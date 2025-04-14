#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Git_Clone 
@File    ：Time_Cut.py
@IDE     ：PyCharm 
@Author  ：XCC
@Date    ：2025/4/14 17:07 
@explain : 
'''

import os
import math
from pydub import AudioSegment

file_path = r"D:\数据集\shipsEar_AUDIOS\10__10_07_13_marDeOnza_Sale.wav"
path, filename = os.path.split(file_path) # 分离路径和文件名
fname, ext = os.path.splitext(filename) # 分离文件名和扩展名
print(f'path: {path}, filename: {filename}')
print(f'fname: {fname}, ext: {ext}')

input_music = AudioSegment.from_wav(file_path)
print(input_music)
print(f'音频时长: {len(input_music) / 1000}秒')

# 设置截取时间
start = 0 * 1000
end = len(input_music)
segment_length = 1 * 1000 # 每段音频长度
number = math.ceil((end-start) / segment_length) # 计算截取的音频段数
print(f'number: {number}')
for i in range(number):
    segment_start = start + i * segment_length
    segment_end = min(segment_start + segment_length, end)
    output_music = input_music[segment_start:segment_end]
    if segment_end  - segment_start < segment_length:
        silence_duration = segment_length - (segment_end - segment_start)
        silence = AudioSegment.silent(duration=silence_duration, frame_rate=input_music.frame_rate)
        output_music = output_music + silence
        print(f"第{i + 1}段不足{segment_length / 1000}秒，已补零至{segment_length / 1000}秒")
    output_path = os.path.join(r"D:\数据集\Dataset_Cut", f"{fname}_{i + 1}{ext}")
    output_music.export(output_path, format="wav")
    print(f"已导出第{i+1}段：{output_path}，长度：{len(output_music)/1000}秒")