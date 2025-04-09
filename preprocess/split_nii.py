import os
import cv2
import nibabel as nib
import numpy as np

# 设置输入和输出文件夹
input_source_dir = 'F:\\CrossMoDA\\source_training'  # 输入T1文件夹
input_target_dir = 'F:\\CrossMoDA\\target_training'  # 输入T2文件夹
output_dir = 'F:\\CrossMoDA\\CrossMoDA'  # 输出切片保存文件夹

# 调窗操作
def window_image(image, window_width, window_level):
    """
    调窗函数，适应指定的窗宽和窗位
    """
    min_intensity = window_level - window_width // 2
    max_intensity = window_level + window_width // 2

    # 限制像素值在 min_intensity 和 max_intensity 之间
    image = np.clip(image, min_intensity, max_intensity)

    # 将像素值归一化到 [0, 255]
    image = ((image - min_intensity) / (max_intensity - min_intensity) * 255).astype(np.uint8)

    return image

# 定义处理每个图像文件的函数
def process_image(image_file):
    print(f'Processing {image_file}')
    # 获取对应的标签文件路径
    if 'ceT1' in image_file:
        image_path = os.path.join(input_source_dir, image_file)
    else:
        image_path = os.path.join(input_target_dir, image_file)

    # 读取图像和标签
    image_nii = nib.load(image_path)
    image_data = image_nii.get_fdata()

    # 创建对应图像文件夹
    if 'ceT1' in image_file:
        image_output_dir = os.path.join(output_dir, 'T1', image_file.split('.')[0].split('_')[1])
        os.makedirs(image_output_dir, exist_ok=True)
    else:
        image_output_dir = os.path.join(output_dir, 'T2', image_file.split('.')[0].split('_')[1])
        os.makedirs(image_output_dir, exist_ok=True)


    # 切片并保存
    slice_num = image_data.shape[2]
    start = 0
    end = slice_num
    step = 1
    if slice_num >= 80:
        step = slice_num // 40
    for i in range(start, end, step):  # 假设按 z 轴切片
        image_slice = image_data[:, :, i]

        # 保存切片图像
        image_slice_path = os.path.join(image_output_dir, f'{i}.png')

        # 归一化图像
        if 'ceT1' in image_file:
            normalized_image_slice = window_image(image_slice, window_width=1638, window_level=819)
        else:
            normalized_image_slice = window_image(image_slice, window_width=1580, window_level=790)

        # 保存图像和标签切片
        cv2.imwrite(image_slice_path, normalized_image_slice)

if __name__ == '__main__':

    # 获取所有图像文件路径
    image_files = [f for f in os.listdir(input_source_dir) if f.endswith('ceT1.nii') or f.endswith('ceT1.nii.gz')]

    # 处理图像文件
    for image_file in image_files:
        process_image(image_file)

    print("所有T1图像的切片保存完毕！")

    # 获取所有图像文件路径
    image_files = [f for f in os.listdir(input_target_dir) if f.endswith('hrT2.nii') or f.endswith('hrT2.nii.gz')]
    # 处理图像文件
    for image_file in image_files:
        process_image(image_file)

    print("所有T2图像的切片保存完毕！")
