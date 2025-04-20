#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Git_Clone 
@File    ：train-test-split.py
@IDE     ：PyCharm 
@Author  ：XCC
@Date    ：2025/4/14 19:19 
@explain : 
'''

import os
import shutil
import pandas as pd
import torchaudio
from sklearn.model_selection import train_test_split

def rename_files_with_labels(audio_dir, label_file, output_dir):
    """
    根据标注文件重命名音频文件，将标签添加到文件名开头
    参数:
    audio_dir: 包含原始音频文件的目录
    label_file: 包含标注信息的txt文件路径
    output_dir: 输出目录，如果为None则在原目录重命名
    """
    # 读取标注文件
    labels_df = pd.read_csv(label_file, sep='\t')  # 根据实际分隔符调整

    # 创建输出目录(如果需要)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有音频文件
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]

    # 重命名计数器
    renamed_count = 0
    not_found_count = 0

    for audio_file in audio_files:
        # 从文件名中提取ID
        try:
            file_id = int(audio_file.split('__')[0])

            # 在标注数据中查找对应的标签
            label_row = labels_df[labels_df['ids'] == file_id]

            if not label_row.empty:
                # 获取标签
                label = label_row['label'].values[0]

                # 构建新文件名
                new_filename = f"{label}_{audio_file}"

                # 重命名或复制文件
                if output_dir:
                    shutil.copy(
                        os.path.join(audio_dir, audio_file),
                        os.path.join(output_dir, new_filename)
                    )
                else:
                    os.rename(
                        os.path.join(audio_dir, audio_file),
                        os.path.join(audio_dir, new_filename)
                    )

                renamed_count += 1
                print(f"已重命名: {audio_file} -> {new_filename}")
            else:
                print(f"警告: 未找到ID为{file_id}的标签信息")
                not_found_count += 1

        except (IndexError, ValueError) as e:
            print(f"错误: 无法处理文件 {audio_file} - {str(e)}")
            not_found_count += 1

    print(f"处理完成!")
    print(f"已重命名: {renamed_count} 个文件")
    print(f"未找到标签: {not_found_count} 个文件")

def train_test_split_by_class(base_dir, class_names, test_ratio=0.2):
    """
    对每个类别文件夹中的音频进行训练集和测试集的划分

    参数:
    base_dir: 包含分类后文件夹的基础目录
    class_names: 类别名称列表
    test_ratio: 测试集比例，默认0.2
    """
    # 创建训练集和测试集目录
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir, exist_ok=True)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir, exist_ok=True)

    # 对每个类别进行划分
    for class_name in class_names:
        class_dir = os.path.join(base_dir, f'Class{class_name}')
        if not os.path.exists(class_dir):
            print(f"警告: 类别目录 {class_dir} 不存在")
            continue

        # 获取该类别的所有音频文件
        audio_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]

        if not audio_files:
            print(f"警告: 类别 {class_name} 中没有音频文件")
            continue

        # 划分训练集和测试集
        train_files, test_files = train_test_split(audio_files, test_size=test_ratio, random_state=42)

        # 复制文件到训练集和测试集目录
        for file in train_files:
            shutil.copy(
                os.path.join(class_dir, file),
                os.path.join(train_dir, f"Class{class_name}_{file}")
            )

        for file in test_files:
            shutil.copy(
                os.path.join(class_dir, file),
                os.path.join(test_dir, f"Class{class_name}_{file}")
            )

        print(f"类别 {class_name}: {len(train_files)} 训练样本, {len(test_files)} 测试样本")

def class5_split(wavs, output_base_path):
    # 定义5个类别及其对应的船舶类型
    class_definitions = {
        'A': ['Fishing', 'Trawler', 'Mussel', 'Tugboat', 'Dredger'],
        'B': ['Motorboat', 'Pilot', 'Sailboat'],
        'C': ['Passenger', 'Ferry'],
        'D': ['OceanLiner', 'Ro-ro'],
        'E': ['Background', 'Noise']
    }
    # 创建输出目录结构
    output_dir = os.path.join(output_base_path, 'shipsear_with_split-5class')
    if not os.path.exists(output_dir):
        for class_name in class_definitions.keys():
            os.makedirs(os.path.join(output_dir, f'Class{class_name}'), exist_ok=True)
    # 获取所有音频文件
    wavs = []
    if os.path.isdir(wavs_path):
        wavs = os.listdir(wavs_path)
    else:
        print(f"错误: {wavs_path} 不是有效目录")
        return
    # 统计各类别文件数量
    class_counts = {class_name: 0 for class_name in class_definitions.keys()}
    unclassified = 0
    # 分类文件
    for wav in wavs:
        if not wav.endswith('.wav'):
            continue

        # 提取船舶类型（文件名的第一部分）
        try:
            ship_type = wav.split('_')[0]
            assigned = False

            # 判断船舶类型属于哪个类别
            for class_name, ship_types in class_definitions.items():
                for ship_pattern in ship_types:
                    if ship_pattern.lower() in ship_type.lower():
                        # 复制文件到相应类别目录
                        shutil.copy(
                            os.path.join(wavs_path, wav),
                            os.path.join(output_dir, f'Class{class_name}', wav)
                        )
                        class_counts[class_name] += 1
                        assigned = True
                        break
                if assigned:
                    break
            if not assigned:
                print(f"未分类: {wav} - 船舶类型 '{ship_type}' 不在定义的类别中")
                unclassified += 1

        except Exception as e:
            print(f"处理 {wav} 时出错: {str(e)}")
            unclassified += 1

    # 输出分类统计信息
    print("\n分类完成！统计信息:")
    for class_name in class_definitions.keys():
        print(f"Class {class_name}: {class_counts[class_name]} 个文件")
    print(f"未分类: {unclassified} 个文件")

    # 进一步对每个类别进行训练/测试集划分
    train_test_split_by_class(output_dir, class_definitions.keys())

    return output_dir

def class9_split(wavs):
    train_files, test_files = [], []
    for wav in wavs:
        # E.G., Tugboat_15__10_07_13_radaUno_Pasa.wav
        if wav.split('.')[-1] != 'wav':
            continue
        if int(wav.split('_')[1]) in [80,93,94,96,73,74,76,21,26,33,39,45,51,52,70,77,79,46,47,49,66,81,82,84,85,86,88,
                                      89,90,91,16,22,23,25,69,6,7,8,10,11,12,14,17,32,34,36,38,40,41,43,53,54,59,60,61,
                                      63,64,67,18,19,58,37,56,68]:
            train_files.append(wav)
        elif int(wav.split('_')[1]) in [95,75,27,50,72,48,83,87,92,24,71,9,13,35,42,55,62,65,20,78,57]:
            test_files.append(wav)
        else:
            print(wav,' is excluded!')
    return train_files, test_files

def class12_split(wavs):
    train_files, test_files = [], []
    for wav in wavs:
        # E.G., Tugboat_15__10_07_13_radaUno_Pasa.wav
        if wav.split('.')[-1] != 'wav':
            continue
        if int(wav.split('_')[1]) in [80, 94, 96, 73, 74, 76, 27, 33, 39, 45, 50, 52, 70, 72, 77, 79, 47, 48, 49,
                                      82, 83, 84, 86, 87, 89, 90, 91, 92, 16, 23, 25, 69, 71, 6, 8, 9, 10, 11, 12,
                                      13, 17, 32, 34, 36, 38, 41, 42, 53, 55, 59, 60, 62, 63, 64, 65, 67, 29, 18,
                                      19, 58, 37, 57, 68, 15]:
                train_files.append(wav)
        elif int(wav.split('_')[1]) in [93, 95, 75, 21, 26, 51, 46, 66, 81, 85, 88, 22, 24, 7, 14, 35, 40, 43, 54,
                                        61, 30, 20, 78, 56, 31]:
            test_files.append(wav)
        else:
            print(wav, ' is excluded!')
    return train_files, test_files

if __name__ == '__main__':
    ###################### 文件重命名处理 ######################
    # audio_directory = r"D:\数据集\ShipEar"
    # labels_file = r"D:\数据集\ShipEar\meta_info.txt"
    # output_directory = r"D:\数据集\ShipEar-Class"
    # rename_files_with_labels(audio_directory, labels_file, output_directory)

    wavs_path = 'D:\数据集\ShipEar-Rename'
    wavs = os.listdir(wavs_path)

    ###################### 5-class train-test-split ######################


    ###################### 9-class train-test-split ######################
    # train9_files, test9_files = class9_split(wavs)
    # out9_path = 'D:\数据集\ShipEar-train-test-split\shipsear_with_split-9class'
    # if not os.path.exists(out9_path):
    #     os.makedirs(os.path.join(out9_path,'train'))
    #     os.makedirs(os.path.join(out9_path,'test'))
    # for x in train9_files:
    #     shutil.copy(os.path.join(wavs_path,x), os.path.join(out9_path,'train',x))
    # for x in test9_files:
    #     shutil.copy(os.path.join(wavs_path,x), os.path.join(out9_path,'test',x))

    ###################### 12-class train-test-split ######################
    # train12_files, test12_files = class12_split(wavs)
    # out12_path = 'D:\数据集\ShipEar-train-test-split\shipsear_with_split-12class'
    # if not os.path.exists(out12_path):
    #     os.makedirs(os.path.join(out12_path,'train'), exist_ok=True)
    #     os.makedirs(os.path.join(out12_path,'test'), exist_ok=True)
    # for x in train12_files:
    #     shutil.copy(os.path.join(wavs_path,x), os.path.join(out12_path,'train',x))
    # for x in test12_files:
    #     shutil.copy(os.path.join(wavs_path,x), os.path.join(out12_path,'test',x))
    #
    # # Cut clips for Trawler_28
    # name = 'Trawler_28__19_07_13_NuevoRiaAldan.wav'
    # waveform, sample_rate = torchaudio.load(os.path.join(wavs_path,name))
    # # print(f"数据类型: {waveform.dtype}") # torch.float32
    # # print(f'waveform: {waveform.size()}, sample_rate: {sample_rate}') # waveform.size() 显示的是张量维度 [通道数, 采样点数]
    # duration = waveform.size(1) / sample_rate
    # start_time = 125
    #
    # part1 = waveform[:, 15 * sample_rate:start_time * sample_rate]
    # part2 = waveform[:, start_time * sample_rate:]
    #
    # # Save as WAV file
    # torchaudio.save(os.path.join(out12_path,'train',name.replace('.wav','_train.wav')), part1, sample_rate, bits_per_sample=16)
    # torchaudio.save(os.path.join(out12_path,'test',name.replace('.wav','_test.wav')), part2, sample_rate, bits_per_sample=16)