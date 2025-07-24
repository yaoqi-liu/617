
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from torch import nn

from nnunet.evaluation.region_based_evaluation import evaluate_regions, get_breast_regions
from nnunet.training.dataloading.dataset_loading_breast import unpack_dataset
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.loss_functions.dice_loss import DC_and_BCE_loss, get_tp_fp_fn_tn, SoftDiceLoss
from nnunet.training.network_training.MAML3_channelTrainerV2_breast import MAML3_channelTrainerV2_breast
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda

from torch.cuda.amp import autocast


class MAML3_channelTrainerV2BreastRegions_BN(MAML3_channelTrainerV2_breast):
    def initialize_network(self):
        """inference_apply_nonlin to sigmoid"""
        super().initialize_network()
        self.network.inference_apply_nonlin = nn.Sigmoid()


class MAML3_channelTrainerV2BreastRegions(MAML3_channelTrainerV2_breast):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.regions = get_breast_regions()
        self.regions_class_order = (1,)
        self.loss = DC_and_BCE_loss({}, {'batch_dice': False, 'do_bg': True, 'smooth': 0})

    def process_plans(self, plans):
        super().process_plans(plans)
        """
        The network has as many outputs as we have regions
        """
        self.num_classes = len(self.regions)

    def initialize_network(self):
        """inference_apply_nonlin to sigmoid"""
        super().initialize_network()
        self.network.inference_apply_nonlin = nn.Sigmoid()

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None): # True
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            # self.ds_loss_weights = weights
            ################################## yao
            self.ds_loss_weights = weights[0] # None -> 0.53333333

            for i in range(self.num_input_channels + 1):
                self.ds_loss_weights = np.append(self.ds_loss_weights, weights)
            # print('################', self.ds_loss_weights)
            ################################## yao
            
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################
            # ..../nnUNet_preprocessed/Task083_debugBraTS2020/nnUNetData_plans_v2.1_stage0'
            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val, self.dl_ts = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data) # '.../nnUNet_preprocessed/Task083_debugBraTS2020/nnUNetData_plans_v2.1_stage0'
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")
                print(self.data_aug_params)
                self.tr_gen, self.val_gen, self.ts_gen = get_moreDA_augmentation(self.dl_tr, self.dl_val, self.dl_ts,
                                                                    self.data_aug_params[
                                                                        'patch_size_for_spatialtransform'],
                                                                    self.data_aug_params,
                                                                    deep_supervision_scales=self.deep_supervision_scales,
                                                                    regions=self.regions)
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("TEST KEYS:\n %s" % (str(self.dataset_ts.keys())),
                                    also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: int = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = False):

        super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)


    def run_online_evaluation(self, output, target):
        output = output[0]
        target = target[0]
        with torch.no_grad():
            out_sigmoid = torch.sigmoid(output)
            out_sigmoid = (out_sigmoid > 0.5).float()

            if self.threeD:
                axes = (0, 2, 3, 4)
            else:
                axes = (0, 2, 3)

            tp, fp, fn, _ = get_tp_fp_fn_tn(out_sigmoid, target, axes=axes)

            tp_hard = tp.detach().cpu().numpy()
            fp_hard = fp.detach().cpu().numpy()
            fn_hard = fn.detach().cpu().numpy()

            self.online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
            self.online_eval_tp.append(list(tp_hard))
            self.online_eval_fp.append(list(fp_hard))
            self.online_eval_fn.append(list(fn_hard))

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """ tr True False  val False True
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)


            ##################################### yao
            if self.epoch % 2:
                tmp_target = [target[0],]

                for i in range(self.num_input_channels + 1):
                    tmp_target += target
                f_target = tmp_target
            else:
                tmp_target = [target[0],]

                for i in range(self.num_input_channels + 1):
                    tmp_target += target
                m_target = tmp_target
            #####################################

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
#################################################################################################
                if self.epoch % 2:
                    f_output = self.network(data)
                    l = self.loss(f_output, f_target)
                else:
                    m_output = self.network(data)
                    l = self.loss(m_output, m_target)

#################################################################################################



            if do_backprop: # True
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
#################################################################################################
            if self.epoch % 2:
                f_output = self.network(data)
                l = self.loss(f_output, f_target) 
            else:
                m_output= self.network(data)
                l = self.loss(m_output, m_target)
            # l = self.loss(f_output, f_target) + self.loss(m_output, m_target)

#################################################################################################

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            if not self.epoch % 2:
                f_output  = self.network(data)
            self.run_online_evaluation(f_output, target)

        del data
        del target

        return l.detach().cpu().numpy()

class MAML3_channelTrainerV2BreastRegions_Dice(MAML3_channelTrainerV2BreastRegions):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.loss = SoftDiceLoss(apply_nonlin=torch.sigmoid, **{'batch_dice': False, 'do_bg': True, 'smooth': 0})
