# CycleGAN-CrossMoDA2022
基于CycleGAN的MRI中T1模态和T2模态图像翻译

step 0：数据集下载：https://zenodo.org/records/4662239#.YmF5tIVBxPY

step 1：使用preprocess切分预处理数据集，先使用split_nii.py切分nii数据成png，后使用split_dataset.py分组数据集为训练集和测试集，保存在该文件夹的txt文件中

step 2：使用train.py训练，训练曲线存储在runs，训练过程保存的图像翻译图片保存在images

step 3：使用test.py测试，图像结果存储在result

step 4：使用result/calculate.py测试fid图像生成指标分数

可以在我的知乎和我讨论：https://www.zhihu.com/people/ou-yang-4-72-1
