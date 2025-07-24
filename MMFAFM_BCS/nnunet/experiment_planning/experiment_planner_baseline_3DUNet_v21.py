#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from copy import deepcopy

import numpy as np
from nnunet.experiment_planning.common_utils import get_pool_and_conv_props
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet import ExperimentPlanner
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.paths import *


class ExperimentPlanner3D_v21(ExperimentPlanner):
    """
    Combines ExperimentPlannerPoolBasedOnSpacing and ExperimentPlannerTargetSpacingForAnisoAxis

    We also increase the base_num_features to 32. This is solely because mixed precision training with 3D convs and
    amp is A LOT faster if the number of filters is divisible by 8
    """
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner3D_v21, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnUNetData_plans_v2.1"
        self.plans_fname = join(self.preprocessed_output_folder,
                                "nnUNetPlansv2.1_plans_3D.pkl")
        self.unet_base_num_features = 32

    def get_target_spacing(self):
        """
        per default we use the 50th percentile=median for the target spacing. Higher spacing results in smaller data
        and thus faster and easier training. Smaller spacing results in larger data and thus longer and harder training

        For some datasets the median is not a good choice. Those are the datasets where the spacing is very anisotropic
        (for example ACDC with (10, 1.5, 1.5)). These datasets still have examples with a spacing of 5 or 6 mm in the low
        resolution axis. Choosing the median here will result in bad interpolation artifacts that can substantially
        impact performance (due to the low number of slices).
        """
        spacings = self.dataset_properties['all_spacings']
        sizes = self.dataset_properties['all_sizes']

        target = np.percentile(np.vstack(spacings), self.target_spacing_percentile, 0)

        # This should be used to determine the new median shape. The old implementation is not 100% correct.
        # Fixed in 2.4
        # sizes = [np.array(i) / target * np.array(j) for i, j in zip(spacings, sizes)]

        target_size = np.percentile(np.vstack(sizes), self.target_spacing_percentile, 0)
        target_size_mm = np.array(target) * np.array(target_size)
        # we need to identify datasets for which a different target spacing could be beneficial. These datasets have
        # the following properties:
        # - one axis which much lower resolution than the others
        # - the lowres axis has much less voxels than the others
        # - (the size in mm of the lowres axis is also reduced)
        worst_spacing_axis = np.argmax(target)
        other_axes = [i for i in range(len(target)) if i != worst_spacing_axis]
        other_spacings = [target[i] for i in other_axes]
        other_sizes = [target_size[i] for i in other_axes]

        has_aniso_spacing = target[worst_spacing_axis] > (self.anisotropy_threshold * max(other_spacings))
        has_aniso_voxels = target_size[worst_spacing_axis] * self.anisotropy_threshold < min(other_sizes)
        # we don't use the last one for now
        #median_size_in_mm = target[target_size_mm] * RESAMPLING_SEPARATE_Z_ANISOTROPY_THRESHOLD < max(target_size_mm)

        if has_aniso_spacing and has_aniso_voxels:
            spacings_of_that_axis = np.vstack(spacings)[:, worst_spacing_axis]
            target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 10)
            # don't let the spacing of that axis get higher than the other axes
            if target_spacing_of_that_axis < max(other_spacings):
                target_spacing_of_that_axis = max(max(other_spacings), target_spacing_of_that_axis) + 1e-5
            target[worst_spacing_axis] = target_spacing_of_that_axis
        return target

    """生成3d_fullres配置"""
    def get_properties_for_stage(self, current_spacing, original_spacing, original_shape, num_cases,
                                 num_modalities, num_classes):
        """
        ExperimentPlanner configures pooling so that we pool late. Meaning that if the number of pooling per axis is
        (2, 3, 3), then the first pooling operation will always pool axes 1 and 2 and not 0, irrespective of spacing.
        This can cause a larger memory footprint, so it can be beneficial to revise this.

        Here we are pooling based on the spacing of the data.

        """
        new_median_shape = np.round(original_spacing / current_spacing * original_shape).astype(int) # 根据原始间距、当前间距和原始形状计算新的中位数形状
        dataset_num_voxels = np.prod(new_median_shape) * num_cases # 根据新的中位数形状 和 图像 数量计算数据集的体素总数

        # the next line is what we had before as a default. The patch size had the same aspect ratio as the median shape of a patient. We swapped t
        # input_patch_size = new_median_shape

        # compute how many voxels are one mm
        input_patch_size = 1 / np.array(current_spacing) # 计算每毫米 对应的体素数量

        # normalize voxels per mm
        input_patch_size /= input_patch_size.mean() # 对每毫米对应的体素数量 进行归一化处理

        # create an isotropic patch of size 512x512x512mm
        input_patch_size *= 1 / min(input_patch_size) * 512  # to get a starting value
        input_patch_size = np.round(input_patch_size).astype(int)

        # clip it to the median shape of the dataset because patches larger then that make not much sense
        input_patch_size = [min(i, j) for i, j in zip(input_patch_size, new_median_shape)]
        # 输入的图像间距（spacing）、输入块大小（patch_size）、特征图最小边长（min_feature_map_size）和最大池化次数（max_numpool），计算网络在每个轴上的池化次数、池化操作的核大小、卷积操作的核大小，同时对输入块大小进行调整，确保其满足一定的可整除性条件。
        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, shape_must_be_divisible_by = \
            get_pool_and_conv_props(current_spacing, input_patch_size,self.unet_featuremap_min_edge_length,self.unet_max_numpool)

        # we compute as if we were using only 30 feature maps. We can do that because fp16 training is the standard
        # now. That frees up some space. The decision to go with 32 is solely due to the speedup we get (non-multiples
        # of 8 are not supported in nvidia amp
        # FP16 训练：在深度学习中，FP16（半精度浮点数）训练已经成为一种标准做法。相比于 FP32（单精度浮点数），FP16 使用更少的内存，从而可以节省显存空间，这使得在计算时可以假设使用更少的特征图（这里假设使用 30 个特征图）。
        # NVIDIA AMP 与特征图数量：NVIDIA 自动混合精度（AMP）技术在训练过程中会根据硬件和数据类型自动调整精度以提高训练速度。在使用 AMP 时，特征图的数量通常需要是 8 的倍数，这里选择 32 可能是为了获得更好的训练速度。)
        ref = Generic_UNet.use_this_for_batch_size_computation_3D * self.unet_base_num_features / Generic_UNet.BASE_NUM_FEATURES_3D # 计算一个参考值
        here = Generic_UNet.compute_approx_vram_consumption(new_shp, network_num_pool_per_axis, # 近似计算在给定网络配置下的显存（VRAM）消耗量
                                                            self.unet_base_num_features,
                                                            self.unet_max_num_filters, num_modalities,
                                                            num_classes,
                                                            pool_op_kernel_sizes, conv_per_stage=self.conv_per_stage)
        # 使用一个 while 循环来持续调整输入数据的形状，直到显存消耗 here 小于或等于参考值 ref。在每次循环中，会选择一个轴来减小输入数据的尺寸，然后重新计算池化和卷积的属性，以及更新显存消耗的估算值。
        while here > ref:
            axis_to_be_reduced = np.argsort(new_shp / new_median_shape)[-1]

            tmp = deepcopy(new_shp)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
            _, _, _, _, shape_must_be_divisible_by_new = \
                get_pool_and_conv_props(current_spacing, tmp,
                                        self.unet_featuremap_min_edge_length,
                                        self.unet_max_numpool,
                                        )
            new_shp[axis_to_be_reduced] -= shape_must_be_divisible_by_new[axis_to_be_reduced]

            # we have to recompute numpool now:
            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, \
            shape_must_be_divisible_by = get_pool_and_conv_props(current_spacing, new_shp,
                                                                 self.unet_featuremap_min_edge_length,
                                                                 self.unet_max_numpool,
                                                                 )

            here = Generic_UNet.compute_approx_vram_consumption(new_shp, network_num_pool_per_axis,
                                                                self.unet_base_num_features,
                                                                self.unet_max_num_filters, num_modalities,
                                                                num_classes, pool_op_kernel_sizes,
                                                                conv_per_stage=self.conv_per_stage)
            #print(new_shp)
        #print(here, ref)

        input_patch_size = new_shp

        batch_size = Generic_UNet.DEFAULT_BATCH_SIZE_3D  # This is what wirks with 128**3  batch_size为2
        batch_size = int(np.floor(max(ref / here, 1) * batch_size))

        # check if batch size is too large 检查和调整批量大小（batch_size），确保其在合理范围内
        max_batch_size = np.round(self.batch_size_covers_max_percent_of_dataset * dataset_num_voxels /
                                  np.prod(input_patch_size, dtype=np.int64)).astype(int)
        max_batch_size = max(max_batch_size, self.unet_min_batch_size)
        batch_size = max(1, min(batch_size, max_batch_size))

        do_dummy_2D_data_aug = (max(input_patch_size) / input_patch_size[0]) > self.anisotropy_threshold # False

        plan = {
            'batch_size': batch_size,
            'num_pool_per_axis': network_num_pool_per_axis,
            'patch_size': input_patch_size,
            'median_patient_size_in_voxels': new_median_shape,
            'current_spacing': current_spacing,
            'original_spacing': original_spacing,
            'do_dummy_2D_data_aug': do_dummy_2D_data_aug,
            'pool_op_kernel_sizes': pool_op_kernel_sizes,
            'conv_kernel_sizes': conv_kernel_sizes,
        }
        return plan
