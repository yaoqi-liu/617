from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np

if __name__ == '__main__':
    input_file = '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/A2FSeg-main/data/nnUNet_preprocessed/Task131_DCET2Reg/nnUNetPlansv2.1_plans_3D.pkl'
    output_file = '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/A2FSeg-main/data/nnUNet_preprocessed/Task131_DCET2Reg/nnUNetPlansv2.1_batch_01_plans_3D.pkl'
    a = load_pickle(input_file)
    # a['plans_per_stage'][0]['batch_size'] = int(np.floor(6 / 9 * a['plans_per_stage'][0]['batch_size']))
    a['plans_per_stage'][0]['batch_size'] = 1 #把nnunet预处理后的pkl文件的batch_size 改为1
    save_pickle(a, output_file)