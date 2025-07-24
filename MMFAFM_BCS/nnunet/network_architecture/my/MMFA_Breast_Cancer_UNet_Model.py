import sys

from copy import deepcopy
from nnunet.utilities.nd_softmax import sigmid_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.my.neural_network3_breast import SegmentationNetwork  # 使用自定义的neural_network3_breast 进行滑动窗口预测
import torch.nn.functional
from nnunet.network_architecture.my.CMFA import CMFA # 多模态特征对齐融合模块

"""它将卷积层（Conv）、丢弃层（Dropout）、归一化层（Norm）和非线性激活层（Nonlin）组合在一起。这个类的设计目的是修复之前版本中无论指定何种非线性激活函数都使用 LeakyReLU 的问题"""
class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))

"""改变了以上函数的 先后 IN 和 relu的问题"""
class ConvDropoutNonlinNorm(ConvDropoutNormNonlin):
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.instnorm(self.lrelu(x))


class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):

        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        return self.blocks(x)


def print_module_training_status(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.Dropout3d) or \
            isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout) or isinstance(module, nn.InstanceNorm3d) \
            or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.InstanceNorm1d) \
            or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module,
                                                                                                      nn.BatchNorm1d):
        print(str(module), module.training)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)


"""编码器 + 解码器"""
class Generic_UNet(nn.Module):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=sigmid_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False):

        super(Generic_UNet, self).__init__()
        # depending on the loss, we do not hard code a nonlinearity into the architecture. To aggregate predictions
        # during inference, we need to apply the nonlinearity, however. So it is important to let the newtork know what
        # to apply in inference. For the most part this will be softmax
        self.inference_apply_nonlin = lambda x: x  # softmax_helper

        # This is for saving a gaussian importance map for inference. It weights voxels higher that are closer to the
        # center. Prediction at the borders are often less accurate and are thus downweighted. Creating these Gaussians
        # can be expensive, so it makes sense to save and reuse them.
        self._gaussian_3d = self._patch_size_for_gaussian_3d = None
        self._gaussian_2d = self._patch_size_for_gaussian_2d = None

        self.convolutional_upsampling = convolutional_upsampling # True
        self.convolutional_pooling = convolutional_pooling # True
        self.upscale_logits = upscale_logits # False
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d: # that
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []

        output_features = base_num_features
        input_features = input_channels

        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))

            output_features = min(output_features, self.max_num_features)

        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling: # False
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
                                            1, 1, 0, 1, 1, seg_output_use_bias))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            # self.apply(print_module_training_status)

    def forward(self, x):
        skips = [] # (2,32,80,192,160),(2,64,40,96,80),(2,128,20,48,40),(2,256,10,24,20),(2,320,5,12,10)
        seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1): # 不要最后一层的bottle_neck卷积
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:  # False
                x = self.td[d](x)

        x = self.conv_blocks_context[-1](x) # bottle_neck
        ##############################################################################
        feature = [x, ]
        ##############################################################################

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)

            ##############################################################################
            feature.append(x)
            ##############################################################################

            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

            if u == len(self.tu) - 1:
                feature_output = x

        # ms_outputs (B,32,80,192,160) (B,1,80,192,160) (B,1,40,96,80)  (B,1,20,48,40)  (B,1,10,24,20)  (B,1,5,12,10)
        # feature  (B,320,5,6,5) (B,320,5,12,10) (B,256,10,24,20)  (B,128,20,48,40)  (B,64,40,96,80)  (B,32,80,192,160)
        return tuple([feature_output, seg_outputs[-1]] + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1],seg_outputs[:-1][::-1])]), feature

##############################################################################
"""特征融合模块 deoder"""


class Generic_UNet_Decoder(nn.Module):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=sigmid_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False):
        """
        basically more flexible than v1, architecture is the same

        Does this look complicated? Nah bro. Functionality > usability

        This does everything you need, including world peace.

        Questions? -> f.isensee@dkfz.de
        """
        super(Generic_UNet_Decoder, self).__init__()
        # depending on the loss, we do not hard code a nonlinearity into the architecture. To aggregate predictions
        # during inference, we need to apply the nonlinearity, however. So it is important to let the newtork know what
        # to apply in inference. For the most part this will be softmax
        self.inference_apply_nonlin = lambda x: x  # softmax_helper

        # This is for saving a gaussian importance map for inference. It weights voxels higher that are closer to the
        # center. Prediction at the borders are often less accurate and are thus downweighted. Creating these Gaussians
        # can be expensive, so it makes sense to save and reuse them.
        self._gaussian_3d = self._patch_size_for_gaussian_3d = None
        self._gaussian_2d = self._patch_size_for_gaussian_2d = None

        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []

        output_features = base_num_features
        input_features = input_channels

        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))

            output_features = min(output_features, self.max_num_features)

        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
                                            1, 1, 0, 1, 1, seg_output_use_bias))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)

        # 模态特征融合模块
        self.modality_feature_align_modules = []
        self.modality_feature_align_modules.append(CMFA(base_num_features,base_num_features,False))
        self.modality_feature_align_modules.append(CMFA(2*base_num_features,2*base_num_features,False))
        self.modality_feature_align_modules.append(CMFA(4*base_num_features,4*base_num_features,False))
        self.modality_feature_align_modules.append(CMFA(8*base_num_features,8*base_num_features,False))
        self.modality_feature_align_modules.append(CMFA(16*base_num_features,16*base_num_features,False))
        self.modality_feature_align_modules.append(CMFA(20*base_num_features,20*base_num_features,False))
        self.modality_feature_align_modules = nn.ModuleList(self.modality_feature_align_modules)

    def forward(self, skip):

        ######特征对齐模块#########
        skips = [] # 单模态的解码器特征对齐融合 # [(2,320,5,6,5),(2,256,5,12,10),(2,128,10,24,20),(2,64,20,48,40),(2,32,40,96,80),(2,16,80,192,160)]
        DCE_modality_features = skip[0]
        T2_modality_features = skip[1]
        for j in range(len(DCE_modality_features)):
            DCE_conv_output = DCE_modality_features[j]
            T2_conv_output = T2_modality_features[j]
            cmfa = self.modality_feature_align_modules[len(DCE_modality_features)-1-j] # 特征对齐
            out = cmfa(DCE_conv_output,T2_conv_output)
            skips.append(out)

        # skips = [] # [(2,320,10,12,10),(2,256,10,24,20),(2,128,20,48,40),(2,64,40,96,80),(2,32,80,192,160)]
        # for i in range(len(skip[0])): # 对两个模态的encoder的 跳跃连接 进行处理
        #     t = []
        #     for j in range(len(skip)):
        #         t.append(skip[j][i])
        #     skips.append(torch.mean(torch.stack(t, 0), 0)) # [(2, 320, 10, 12, 10),(2, 320, 10, 12, 10)] -> (2, 2, 320, 10, 12, 10)->(2, 320, 10, 12, 10)

        seg_outputs = []


        x = skips[0] # 直接进入编码器模块的encoder输出

        for u in range(len(self.tu)): # 进行残差连接
            x = self.tu[u](x) # 把残差
            x = torch.cat((x, skips[u + 1]), dim=1) # 残差与decoder输出 拼接
            x = self.conv_blocks_localization[u](x) # decoder feature

            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x))) #把32 64 128 256 320等通道conv->1 ->  sigmod 为了做DS

            if u == len(self.tu) - 1: # 最后一个decoder的输出
                feature_output = x
        # 模态融合的特征（2，32，80，192，160） ,模态融合的DS（2，1，80，192，160） + (2,1,40,96,80) (2,1,20,48,40) (2,1,10,24,20) (2,1,5,12,10)
        return tuple([feature_output, seg_outputs[-1]] + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1],
                                                                               seg_outputs[:-1][::-1])])


"""diy多模态的分割网络"""
class MMFA_Breast_Cancer_UNet_Model(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2  # 3d batch
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)  # 3d patch
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30  # 3d feature 数量
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)  # 2d patch
    BASE_NUM_FEATURES_2D = 30  # 2d feature 数量
    DEFAULT_BATCH_SIZE_2D = 50  # 2d batch
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(self, modality_num, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=sigmid_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False, # False, True, True
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False): # True

        super(MMFA_Breast_Cancer_UNet_Model, self).__init__()
        self.inference_apply_nonlin = lambda x: x  # sigmid_helper
        self._gaussian_3d = self._patch_size_for_gaussian_3d = None
        self._gaussian_2d = self._patch_size_for_gaussian_2d = None
        self.modality_num = modality_num  # 模态的数量
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.conv_op = conv_op

        self.convolutional_upsampling = convolutional_upsampling # True
        self.convolutional_pooling = convolutional_pooling # True
        self.upscale_logits = upscale_logits

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes  # 分类的数量，即分割任务中的类别数
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        # 编码器
        self.modality_specific_models = []
        for i in range(modality_num):
            self.modality_specific_models.append(
                Generic_UNet(1, base_num_features, num_classes, num_pool, num_conv_per_stage,  # 一个模态对应一个encoder
                             feat_map_mul_on_downscale, conv_op,
                             norm_op, norm_op_kwargs,
                             dropout_op, dropout_op_kwargs,
                             nonlin, nonlin_kwargs, deep_supervision, dropout_in_localization,
                             final_nonlin, weightInitializer, pool_op_kernel_sizes,
                             conv_kernel_sizes,
                             upscale_logits, convolutional_pooling, convolutional_upsampling,
                             max_num_features, basic_block,
                             seg_output_use_bias))
        # 解码器
        self.fusion_decoder = Generic_UNet_Decoder(1, base_num_features, num_classes, num_pool, num_conv_per_stage,   # 单只有 一个 decoder
                                                   feat_map_mul_on_downscale, conv_op,
                                                   norm_op, norm_op_kwargs,
                                                   dropout_op, dropout_op_kwargs,
                                                   nonlin, nonlin_kwargs, deep_supervision, dropout_in_localization,
                                                   final_nonlin, weightInitializer, pool_op_kernel_sizes,
                                                   conv_kernel_sizes,
                                                   upscale_logits, convolutional_pooling, convolutional_upsampling,
                                                   max_num_features, basic_block,
                                                   seg_output_use_bias)

        if nonlin_kwargs is None:  # False
            self.nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        else:
            self.nonlin_kwargs = nonlin_kwargs

        if dropout_op_kwargs is None:  # False
            self.dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        else:
            self.dropout_op_kwargs = dropout_op_kwargs

        if norm_op_kwargs is None:  # False
            self.norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        else:
            self.norm_op_kwargs = norm_op_kwargs

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}
        self.conv_kwargs['kernel_size'] = 3
        self.conv_kwargs['padding'] = 1

        self.modality_aware_modules = []
        for i in range(modality_num):  # 为每个模态创建一个模态感知模块
            self.modality_aware_modules.append(
                nn.Sequential(conv_op(2 * base_num_features, base_num_features, **self.conv_kwargs),
                              norm_op(base_num_features, **self.norm_op_kwargs),
                              nonlin(**self.nonlin_kwargs),
                              conv_op(base_num_features, base_num_features, **self.conv_kwargs),
                              nn.LeakyReLU()))

        ##############################################################################
        self.sigmod = nn.Sigmoid()
        self.lastconv = []
        self.lastconv.append(nn.Sequential(conv_op(2 * base_num_features, base_num_features, **self.conv_kwargs),
                                           norm_op(base_num_features, **self.norm_op_kwargs),
                                           nonlin(**self.nonlin_kwargs)))

        self.lastconv = nn.ModuleList(self.lastconv)
        ##############################################################################
        # in out k_size stride pad  dilation = 1 表示卷积核元素之间没有额外的间距 groups = 1 表示不进行分组卷积，即所有输入通道都参与卷积计算  bias=seg_output_use_bias=False
        self.output = conv_op(base_num_features, num_classes, 1, 1, 0, 1, 1, seg_output_use_bias)

        self.modality_specific_models = nn.ModuleList(self.modality_specific_models)
        self.modality_aware_modules = nn.ModuleList(self.modality_aware_modules)

    # 乳腺多模态网络 前向传播
    def forward(self, x):
        x = torch.chunk(x, 2, dim=1)  # (B,2,80,192,160) ->( (B,1,80,192,160),(B,1,80,192,160) )
        modality_features = []
        final_outputs = []

        ##############################################################################
        fusion_feature = []
        ##############################################################################
        # 一个模态 一个encoder 提取各自的特征
        for i in range(self.modality_num):
            ##############################################################################
            # ms_outputs (B,32,80,192,160) (B,1,80,192,160) (B,1,40,96,80)  (B,1,20,48,40)  (B,1,10,24,20)  (B,1,5,12,10)
            # feature  (B,320,5,6,5) (B,320,5,12,10) (B,256,10,24,20)  (B,128,20,48,40)  (B,64,40,96,80)  (B,32,80,192,160)
            ms_outputs, feature = self.modality_specific_models[i](x[i])  # 第i个模态图像 对应第i个 unet特征提取网络

            fusion_feature.append(feature)  #单个模态的残差连接特征 2*[(B,320,5,6,5) (B,320,5,12,10) (B,256,10,24,20)  (B,128,20,48,40)  (B,64,40,96,80)  (B,32,80,192,160)]
            ##############################################################################
            modality_features.append(ms_outputs[0])  # [(B,32,80,192,160),(B,32,80,192,160)] # DCE和T2 两个模态通过第一个卷积 1通道 -> 32 通道
            final_outputs += ms_outputs[1:]  # 2*[(B,1,80,192,160) (B,1,40,96,80)  (B,1,20,48,40)  (B,1,10,24,20)  (B,1,5,12,10)] DCE和T2 两个模态下采样嘛
        ##############################################################################

        # 把残差的特征 在decoder进行融合
        ms_outputs = self.fusion_decoder(fusion_feature)  #  (B,32,80,192,160)  , (B,1,80,192,160) (B,1,40,96,80)  (B,1,20,48,40)  (B,1,10,24,20) (B,1,5,12,10)
        modality_features.append(ms_outputs[0])  # [(B,32,80,192,160),(B,32,80,192,160)] + (B,32,80,192,160) 两个单模态decoder输出的特征 多模态融合decoder输出的特征
        final_outputs += ms_outputs[1:]  # 3*[(B,32,80,192,160) (B,1,40,96,80)  (B,1,20,48,40)  (B,1,10,24,20)  (B,1,5,12,10)] 拿来后面做DS吧
        # ---------------------------------------begin---------------------------------------
        attention_maps = []
        t = 0
        for i in range(self.modality_num):
            ##############################################################################
            attention_maps.append(self.modality_aware_modules[i](torch.cat([modality_features[t], modality_features[-1]],dim=1)))  # （32+32）64通道 卷积变成 32通道 [(B,32,80,192,160),(B,32,80,192,160)]
            t = t + 1

        ##############################################################################
        attention_maps = self.sigmod(torch.stack(attention_maps, 1))  # (B，2，32，80，192，160)
        output = attention_maps[:, 0] * modality_features[0]  # (B，2，32，80，192，160) -> (B,32,80，192，160) * (B,32,80，192，160)->(B,32,80，192，160)
        for i in range(1, 2):
            output += attention_maps[:, i] * modality_features[i]  # (B,32,80,192,160)
        output = self.lastconv[0](torch.cat((output, modality_features[-1]), dim=1))  # （32+32）64通道 卷积变成 32通道  (B,32,80,192,160)
        ##############################################################################

        output = self.output(output)  # 32通道 -> 1通道 (B,1,80,192,160)
        final_outputs = [output, ] + final_outputs  # (B,1,80,192,160) +  2*[(B,1,80,192,160) (B,1,40,96,80)  (B,1,20,48,40)  (B,1,10,24,20)]
        # return final_outputs
        return final_outputs[0]

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):

        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (
                        npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
        return tmp


if __name__ == '__main__':
    num_input_channels = 2
    base_num_features = 16
    conv_per_stage = 2
    num_classes = 1
    net_numpool = 5
    conv_op = nn.Conv3d
    dropout_op = nn.Dropout3d
    norm_op = nn.InstanceNorm3d
    norm_op_kwargs = {'eps': 1e-5, 'affine': True}
    dropout_op_kwargs = {'p': 0, 'inplace': True}
    net_nonlin = nn.LeakyReLU
    net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
    net_num_pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
    net_conv_kernel_sizes = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    model = MMFA_Breast_Cancer_UNet_Model(num_input_channels, base_num_features, num_classes, net_numpool,
                                    conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                                    net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)
    data = torch.rand(2,2,80,192,160)
    output = model(data)
    print(model)
    print(*(o.shape for o in output))


