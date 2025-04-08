#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Git_Clone 
@File    ：audio_test1.py.py
@IDE     ：PyCharm 
@Author  ：XCC
@Date    ：2025/4/5 13:08 
@explain : 
'''

import librosa
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# array_1, sampling_rate_1 = librosa.load(r"D:\数据集\shipsEar_AUDIOS\30__19_07_13_practico2.wav")
# print(f"array_1: {array_1}")
# print(f"array_1 shape: {array_1.shape}") # (6644996,)
# print(f"sampling_rate: {sampling_rate_1}") # 22050

array_2, sampling_rate_2 = librosa.load(r"D:\数据集\shipsEar_AUDIOS\30__19_07_13_practico2.wav", sr=16000)
print(f"array_2: {array_2}")
print(f"array_2 shape: {array_2.shape}")
print(f"sampling_rate: {sampling_rate_2}")

# array_3, sampling_rate_3 = librosa.load(r"D:\数据集\shipsEar_AUDIOS\30__19_07_13_practico2.wav", offset=20, duration=5)
# print(f"array_3: {array_3}")
# print(f"array_3 shape: {array_3.shape}")
# print(f"sampling_rate: {sampling_rate_3}")

# # 波形显示
# fig_1 = plt.figure(1)
# y = array_2
# x = np.arange(len(y)) / sampling_rate_2
# plt.plot(x, y)
# plt.xlabel('Time')
# plt.title("Waveform")
# plt.show()

# 波形显示
# fig_2, ax=plt.subplots(3, 1, constrained_layout=True)
# librosa.display.waveshow(array_1, sr=sampling_rate_1, ax=ax[0])
# librosa.display.waveshow(array_2, sr=sampling_rate_2, ax=ax[1])
# ax[2].set(xlim=(20, 25))
# librosa.display.waveshow(array_3, sr=sampling_rate_3, ax=ax[2], offset=20)
# fig_2.suptitle("Waveform 波形图", fontsize=16)
# ax[0].set_xlabel("Time")
# ax[1].set_xlabel("Time")
# ax[2].set_xlabel("Time")
# ax[0].set_ylabel("Amplitude")
# ax[1].set_ylabel("Amplitude")
# ax[2].set_ylabel("Amplitude")
# plt.show()

# 波形图
# plt.figure(figsize=(12, 8))
# plt.ylabel("Amplitude")
# librosa.display.waveshow(array, sr=sampling_rate)
# plt.show()

# 静音消除（前后部分）
# array_trim, index = librosa.effects.trim(array_2)
# fig, ax = plt.subplots(2, 1, constrained_layout=True)
# librosa.display.waveshow(array_2, sr=sampling_rate_2, ax=ax[0])
# librosa.display.waveshow(array_trim, sr=sampling_rate_2, ax=ax[1])
# plt.title("Waveform with Silence Trimming")
# ax[0].set_xlabel("Time")
# ax[1].set_xlabel("Time")
# ax[0].set_ylabel("Amplitude")
# ax[1].set_ylabel("Amplitude")
# plt.show()

# 静音消除（中间部分）


# 频谱图
# dft_input = array[:]
# # 计算DFT
# window = np.hanning(len(dft_input))
# windowed_input = dft_input * window
# dft = np.fft.rfft(windowed_input)
# # 计算频谱的幅值
# amplitude = np.abs(dft)
# amplitude_db = librosa.amplitude_to_db(amplitude, ref=np.max)
# # 计算每个DFT分量对应的频率值
# frequency = librosa.fft_frequencies(sr=sampling_rate, n_fft=len(dft_input))
# plt.figure().set_figwidth(12)
# plt.plot(frequency, amplitude_db)
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Amplitude (dB)")
# plt.xscale("log")
# plt.show()

# 计算STFT，时频谱
# D = librosa.stft(array)
# S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
# plt.figure().set_figwidth(12)
# librosa.display.specshow(S_db, x_axis="time", y_axis="hz")
# plt.colorbar()
# plt.show()

# 梅尔谱图
# S = librosa.feature.melspectrogram(y=array, sr=sampling_rate, n_mels=128, fmax=8000)
# S_dB = librosa.power_to_db(S, ref=np.max)
# plt.figure().set_figwidth(12)
# librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sampling_rate, fmax=8000)
# plt.colorbar()
# plt.show()