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

import SimpleITK as sitk
import numpy as np
import shutil
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool
from collections import OrderedDict


def create_nonzero_mask(data):
    from scipy.ndimage import binary_fill_holes
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)# 创建 全0 与图像shape一致
    for c in range(data.shape[0]): # 针对每个通道，生成一个布尔型掩码 this_mask，当该通道中的元素不为 0 时，对应位置的值为 True，否则为 False。
        this_mask = data[c] != 0
        nonzero_mask = nonzero_mask | this_mask # 按位或 只要在任意一个通道中某个位置的值不为 0，nonzero_mask 中该位置的值就会变为 True
    nonzero_mask = binary_fill_holes(nonzero_mask) # 对 nonzero_mask 进行孔洞填充操作。该函数会将二进制数组中被 True 区域包围的 False 区域填充为 True
    return nonzero_mask

""" 根绝图像生成的掩码 找 bounding box """
def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

"""根据给定的bbox坐标 裁剪出bbox位置"""
def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]

""" 获取文件名共同前缀  """
def get_case_identifier(case):
    case_identifier = case[0].split("/")[-1].split(".nii")[0][:-5] # [序列1，序列2，序列3，序列4，标签] -> .../images/BraTS20_Training_176_0000.nii' -> BraTS20_Training_176
    return case_identifier # BraTS20_Training_176


def get_case_identifier_from_npz(case):
    case_identifier = case.split("/")[-1][:-4]
    return case_identifier


def load_case_from_list_of_files(data_files, seg_file=None):
    assert isinstance(data_files, list) or isinstance(data_files, tuple), "case must be either a list or a tuple"
    properties = OrderedDict() # 创建有序字典properties
    data_itk = [sitk.ReadImage(f) for f in data_files] # 读取图像文件

    properties["original_size_of_raw_data"] = np.array(data_itk[0].GetSize())[[2, 1, 0]] # 得到第一个图像的 原size numpy格式的(因为尺寸都是一样的)
    properties["original_spacing"] = np.array(data_itk[0].GetSpacing())[[2, 1, 0]] # 得到第一个图像的 原spacing numpy格式的
    properties["list_of_data_files"] = data_files # 保存 图像 路径
    properties["seg_file"] = seg_file # 保存 标签 路径

    properties["itk_origin"] = data_itk[0].GetOrigin() # origin tuple格式
    properties["itk_spacing"] = data_itk[0].GetSpacing() # spacing tuple格式
    properties["itk_direction"] = data_itk[0].GetDirection() # direction  tuple格式

    data_npy = np.vstack([sitk.GetArrayFromImage(d)[None] for d in data_itk]) # 把每个图像转换为numpy格式 在前面新增一个维度，把四张图像再第一个维度堆叠成npy文件 （4，D,W,H）
    if seg_file is not None:
        seg_itk = sitk.ReadImage(seg_file)
        seg_npy = sitk.GetArrayFromImage(seg_itk)[None].astype(np.float32) # 把标签单独 堆叠为一个 npy文件
    else:
        seg_npy = None
    return data_npy.astype(np.float32), seg_npy, properties # 堆叠后的图像npy, 标签npy，properties信息


def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data) # 根据图像 生成掩码
    bbox = get_bbox_from_mask(nonzero_mask, 0) # bbox

    cropped_data = []
    for c in range(data.shape[0]): #遍历每一张图像 根据给定的bbox坐标 裁剪出bbox位置
        cropped = crop_to_bbox(data[c], bbox)
        cropped_data.append(cropped[None])
    data = np.vstack(cropped_data) # 四张各自裁剪后的bbox 再堆叠起来 （按道理四张的bbox应该一样，也一定要一样）

    if seg is not None: # 标签 同上
        cropped_seg = []
        for c in range(seg.shape[0]):
            cropped = crop_to_bbox(seg[c], bbox)
            cropped_seg.append(cropped[None])
        seg = np.vstack(cropped_seg)

    nonzero_mask = crop_to_bbox(nonzero_mask, bbox)[None] # 将新建的 非零掩码 根据给定的边界框 bbox 进行裁剪 大小就和 处理后的标签一致
    if seg is not None: # 当有 标签 时
        seg[(seg == 0) & (nonzero_mask == 0)] = nonzero_label #  通过将特定的 0 值区域标记为 -1，在计算肿瘤区域平均像素值时可以忽略这些无效区域，避免对结果产生干扰
    else:
        nonzero_mask = nonzero_mask.astype(int)
        nonzero_mask[nonzero_mask == 0] = nonzero_label
        nonzero_mask[nonzero_mask > 0] = 0
        seg = nonzero_mask
    return data, seg, bbox


def get_patient_identifiers_from_cropped_files(folder):
    return [i.split("/")[-1][:-4] for i in subfiles(folder, join=True, suffix=".npz")]


class ImageCropper(object):
    def __init__(self, num_threads, output_folder=None):
        """
        This one finds a mask of nonzero elements (must be nonzero in all modalities) and crops the image to that mask.
        In the case of BRaTS and ISLES data this results in a significant reduction in image size
        :param num_threads:
        :param output_folder: whete to store the cropped data
        :param list_of_files:
        """
        self.output_folder = output_folder
        self.num_threads = num_threads

        if self.output_folder is not None:
            maybe_mkdir_p(self.output_folder)

    @staticmethod
    def crop(data, properties, seg=None):
        shape_before = data.shape # 未裁剪bbox的shape
        data, seg, bbox = crop_to_nonzero(data, seg, nonzero_label=-1) #裁剪出（4图像和1标签）bbox区域  通过将特定的 0 值区域标记为 -1，在计算肿瘤区域平均像素值时可以忽略这些无效区域，避免对结果产生干扰
        shape_after = data.shape # 裁剪bbox后的shaoe
        print("before bbox crop:", shape_before, "after bbox crop:", shape_after, "spacing:",
              np.array(properties["original_spacing"]), "\n") # 打印裁剪 bbox 前后的 图像信息

        properties["crop_bbox"] = bbox # 保存每个图像的 bbox 坐标
        properties['classes'] = np.unique(seg)# 保存 处理后的 标签类别 （0，1，2，3）-> (-1,0,1,2,3) 全0背景区域 变成 -1
        seg[seg < -1] = 0 # 避免标签异常值
        properties["size_after_cropping"] = data[0].shape # 保存 裁剪bbox后的 图像size
        return data, seg, properties

    @staticmethod
    def crop_from_list_of_files(data_files, seg_file=None):
        data, seg, properties = load_case_from_list_of_files(data_files, seg_file) # 堆叠后的图像npy, 标签npy，properties信息
        return ImageCropper.crop(data, properties, seg)

    def load_crop_save(self, case, case_identifier, overwrite_existing=False):
        try:
            print(case_identifier) # 打印正在处理的文件前缀 BraTS20_Training_260
            if overwrite_existing or (not os.path.isfile(os.path.join(self.output_folder, "%s.npz" % case_identifier))
                        or not os.path.isfile(os.path.join(self.output_folder, "%s.pkl" % case_identifier))):

                data, seg, properties = self.crop_from_list_of_files(case[:-1], case[-1]) # 分开输入四个模态数据和一个mask，输出模态合一data和mask，数据property

                all_data = np.vstack((data, seg)) # 把裁剪bbox后的图像和标签 堆叠在一块 （4，D,W,H）+ （1，D,W,H）-> (5,D,W,H)
                np.savez_compressed(os.path.join(self.output_folder, "%s.npz" % case_identifier), data=all_data) # 保存为npz文件 BraTS20_Training_117.npz
                with open(os.path.join(self.output_folder, "%s.pkl" % case_identifier), 'wb') as f: # 把更新后的properties 写入
                    pickle.dump(properties, f)
        except Exception as e:
            print("Exception in", case_identifier, ":")
            print(e)
            raise e

    def get_list_of_cropped_files(self):
        return subfiles(self.output_folder, join=True, suffix=".npz")

    def get_patient_identifiers_from_cropped_files(self):
        return [i.split("/")[-1][:-4] for i in self.get_list_of_cropped_files()]

    def run_cropping(self, list_of_files, overwrite_existing=False, output_folder=None):
        """
        also copied ground truth nifti segmentation into the preprocessed folder so that we can use them for evaluation
        on the cluster
        :param list_of_files: list of list of files [[PATIENTID_TIMESTEP_0000.nii.gz], [PATIENTID_TIMESTEP_0000.nii.gz]]
        :param overwrite_existing:
        :param output_folder:
        :return:
        """
        if output_folder is not None:
            self.output_folder = output_folder # '..../nnUNet_cropped_data/Task083_debugBraTS2020'

        output_folder_gt = os.path.join(self.output_folder, "gt_segmentations")# '..../nnUNet_cropped_data/Task083_debugBraTS2020/gt_segmentations'
        maybe_mkdir_p(output_folder_gt)
        for j, case in enumerate(list_of_files): # 枚举 训练+验证+测试集文件与其标签 [[序列1，序列2，序列3，序列4，标签],.....]
            if case[-1] is not None: # 复制标签文件到gt_segmentations文件夹下
                shutil.copy(case[-1], output_folder_gt)

        list_of_args = []
        for j, case in enumerate(list_of_files):# 再枚举 训练+验证+测试集文件与其标签 [[序列1，序列2，序列3，序列4，标签],.....]
            case_identifier = get_case_identifier(case) # 获取文件名共同前缀 'BraTS20_Training_176'
            list_of_args.append((case, case_identifier, overwrite_existing))

        p = Pool(self.num_threads) # 创建多线程一起执行
        p.starmap(self.load_crop_save, list_of_args) # 并行执行 load_crop_save方法
        p.close()
        p.join()

    def load_properties(self, case_identifier):
        with open(os.path.join(self.output_folder, "%s.pkl" % case_identifier), 'rb') as f:
            properties = pickle.load(f)
        return properties

    def save_properties(self, case_identifier, properties):
        with open(os.path.join(self.output_folder, "%s.pkl" % case_identifier), 'wb') as f:
            pickle.dump(properties, f)
