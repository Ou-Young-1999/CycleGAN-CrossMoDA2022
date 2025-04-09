import os
import random

def split_data(files):
    # 确保随机性
    random.seed(3407)
    random.shuffle(files)

    # 计算索引位置
    total_size = len(files)
    train_size = int(total_size * 0.6)

    # 划分数据集
    train_set = files[:train_size]
    test_set = files[train_size:]

    return train_set, test_set

def statistical_dataset(data, modality):
    dataset_size = 0
    root_path = 'F:\\CrossMoDA\\CrossMoDA'
    for case in data:
        case_path = os.path.join(root_path, modality, case)
        sub_path = os.listdir(case_path)
        dataset_size += len(sub_path)

    return dataset_size

def write_txt(data, txt_path):
    # 写入到txt
    with open(txt_path, 'w') as file:
        for case in data[:len(data)-1]:
            file.write(case + '\n')
        file.write(data[-1]) # 最后一行不需要换行
    print(f'writing {txt_path}')

if __name__ == '__main__':
    # modality T1
    root_path = 'F:\\CrossMoDA\\CrossMoDA\\T1'
    files = os.listdir(root_path)
    print(f'T1 modality total size: {len(files)}')
    train_set, test_set = split_data(files)
    print(f'train size: {len(train_set)}')
    print(f'test size: {len(test_set)}')

    train_size = statistical_dataset(train_set, 'T1')
    test_size = statistical_dataset(test_set, 'T1')
    print(f'T1 modality train image size: {train_size}')
    print(f'T1 modality test image size: {test_size}')

    write_txt(train_set, 'trainT1.txt')
    write_txt(test_set, 'testT1.txt')

    # modality T2
    root_path = 'F:\\CrossMoDA\\CrossMoDA\\T2'
    files = os.listdir(root_path)
    print(f'T2 modality total size: {len(files)}')
    train_set, test_set = split_data(files)
    print(f'train size: {len(train_set)}')
    print(f'test size: {len(test_set)}')

    train_size = statistical_dataset(train_set, 'T2')
    test_size = statistical_dataset(test_set, 'T2')
    print(f'T2 modality train image size: {train_size}')
    print(f'T2 modality test image size: {test_size}')

    write_txt(train_set, 'trainT2.txt')
    write_txt(test_set, 'testT2.txt')