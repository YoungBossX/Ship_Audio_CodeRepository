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
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# array_1, sampling_rate_1 = librosa.load(r"E:\数据集\ShipEar\shipsEar_AUDIOS\10__10_07_13_marDeOnza_Sale.wav")
# print(f"array_1: {array_1}")
# print(f"array_1 shape: {array_1.shape}") # (6644996,)
# print(f"sampling_rate: {sampling_rate_1}") # 22050

array_2, sampling_rate_2 = librosa.load(r"E:\数据集\ShipEar\shipsEar_AUDIOS\10__10_07_13_marDeOnza_Sale.wav", sr=16000)
print(f"array_2: {array_2}")
print(f"array_2 shape: {array_2.shape}")
print(f"sampling_rate: {sampling_rate_2}")

# array_3, sampling_rate_3 = librosa.load(r"E:\数据集\ShipEar\shipsEar_AUDIOS\10__10_07_13_marDeOnza_Sale.wav", offset=20, duration=5)
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
# array_trim, index = librosa.effects.trim(array_2, top_db=20)
# fig, ax = plt.subplots(2, 1, constrained_layout=True)
# librosa.display.waveshow(array_2, sr=sampling_rate_2, ax=ax[0])
# ax[0].vlines(index[0] / sampling_rate_2, -0.5, 0.5, colors='r')
# ax[0].vlines(index[1] / sampling_rate_2, -0.5, 0.5, colors='r')
# librosa.display.waveshow(array_trim, sr=sampling_rate_2, ax=ax[1])
# fig.suptitle("Waveform with Silence Trimming")
# ax[0].set_xlabel("Time")
# ax[1].set_xlabel("Time")
# ax[0].set_ylabel("Amplitude")
# ax[1].set_ylabel("Amplitude")
# plt.show()

# 静音消除（中间部分）
# intervals = librosa.effects.split(array_2, top_db=20)
# print("intervals:", intervals)
# array_remix = librosa.effects.remix(array_2, intervals=intervals)
# fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, constrained_layout=True)
# librosa.display.waveshow(array_2, sr=sampling_rate_2, ax=ax[0])
# librosa.display.waveshow(array_remix, sr=sampling_rate_2, ax=ax[1], offset=intervals[0][0]/sampling_rate_2)
# for interval in intervals:
#     ax[0].vlines(interval[0] / sampling_rate_2, -0.5, 0.5, colors='r')
#     ax[0].vlines(interval[1] / sampling_rate_2, -0.5, 0.5, colors='r')
# ax[0].set_xlabel("Time")
# ax[1].set_xlabel("Time")
# ax[0].set_ylabel("Amplitude")
# ax[1].set_ylabel("Amplitude")
# plt.show()

# 预加重波形图
# array_preemp = librosa.effects.preemphasis(array_2)
# fig, ax = plt.subplots(2, 1, constrained_layout=True)
# librosa.display.waveshow(array_2, sr=sampling_rate_2, ax=ax[0])
# librosa.display.waveshow(array_preemp, sr=sampling_rate_2, ax=ax[1])
# ax[0].set_title("Original Waveform")
# ax[1].set_title("Pre-emphasized Waveform")
# ax[0].set_xlabel("Time")
# ax[1].set_xlabel("Time")
# ax[0].set_ylabel("Amplitude")
# ax[1].set_ylabel("Amplitude")
# plt.show()

# 频域表示：STFT
# frame = 25 # 帧长
# hop_length = 10 # 帧移
# win_length = int(frame * sampling_rate_2 / 1000)
# hop_length = int(hop_length * sampling_rate_2 / 1000)
# n_fft = int(2 ** np.ceil(np.log2(win_length)))
# S = np.abs(librosa.stft(array_2, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
# print(f'array_2 length: {len(array_2)}')
# print(f'S : {S.shape}') # (D,N)=(n_fft/2+1, len(array_2)/hop_length)
# fig  = plt.figure()
# librosa.display.specshow(
#     S,
#     sr=sampling_rate_2,
#     hop_length=hop_length,
#     x_axis='time',
#     y_axis='hz',
#     cmap='hot'
# )
# # plt.imshow(S, origin='lower', cmap='hot')
# plt.title('Spectrogram')
# plt.xlabel('Time')
# plt.ylabel('Frequency (Hz)')
# plt.colorbar(format='%+2.0f')
# plt.tight_layout()
# plt.show()

# 清楚的观察高频信息，让数值的动态范围更广
# frame = 25 # 帧长
# hop_length = 10 # 帧移
# win_length = int(frame * sampling_rate_2 / 1000)
# hop_length = int(hop_length * sampling_rate_2 / 1000)
# n_fft = int(2 ** np.ceil(np.log2(win_length)))
# S = np.abs(librosa.stft(array_2, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
# S_dB = librosa.amplitude_to_db(S, ref=np.max)
# D, N = S.shape
# range_D = np.arange(0, D, D / 10)
# range_N = np.arange(0, N, N / 10)
# range_t = range_N * (hop_length / sampling_rate_2 / 1000) # 时间轴
# range_f = range_D * (sampling_rate_2 / n_fft / 1000) # 频率轴（kHz）
# fig  = plt.figure()
# plt.xticks(range_N, [f"{t:.1f}" for t in range_t])
# plt.yticks(range_D, [f"{f:.1f}" for f in range_f])
# print('S shape:', S.shape)
# plt.imshow(S, origin='lower', cmap='hot', aspect='auto')
# plt.colorbar(format='%+2.0f dB')
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (kHz)')
# plt.title('Logarithmic Spectrogram')
# plt.tight_layout()
# plt.show()

# 使用librosa的specshow函数替代imshow以获得更好的频谱图显示
# frame = 25 # 帧长
# hop_length = 10 # 帧移
# win_length = int(frame * sampling_rate_2 / 1000)
# hop_length = int(hop_length * sampling_rate_2 / 1000)
# n_fft = int(2 ** np.ceil(np.log2(win_length)))
# S = np.abs(librosa.stft(array_2, n_fft=n_fft, hop_length=hop_length, win_length=win_length)) # librosa.stft()函数的输出是复数形式，通常希望得到的频谱是实数值，代表信号的幅度信息
# S_dB = librosa.amplitude_to_db(S, ref=np.max)
# librosa.display.specshow(
#     S_dB,
#     sr=sampling_rate_2,
#     hop_length=hop_length,
#     x_axis='time',
#     y_axis='hz',
#     cmap='hot'  # 使用更清晰的颜色映射
# )
# plt.colorbar(format='%+2.0f dB')
# plt.title('Logarithmic Spectrogram')
# plt.tight_layout()
# plt.show()

# 预加重频谱图
# frame = 25 # 帧长
# hop_length = 10 # 帧移
# win_length = int(frame * sampling_rate_2 / 1000)
# hop_length = int(hop_length * sampling_rate_2 / 1000)
# n_fft = int(2 ** np.ceil(np.log2(win_length)))
# array_preemp = librosa.effects.preemphasis(array_2)
# S = np.abs(librosa.stft(array_2, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
# S = librosa.amplitude_to_db(S, ref=np.max)
# S_preemp = np.abs(librosa.stft(array_preemp, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
# S_preemp = librosa.amplitude_to_db(S_preemp, ref=np.max)
# fig, ax = plt.subplots(2, 1, constrained_layout=True)
# img_1 = librosa.display.specshow(S, sr=sampling_rate_2, hop_length=hop_length, x_axis='time', y_axis='hz', cmap='hot', ax=ax[0])
# img_2 = librosa.display.specshow(S_preemp, sr=sampling_rate_2, hop_length=hop_length, x_axis='time', y_axis='hz', cmap='hot', ax=ax[1])
# ax[0].set_title("Original Spectrogram")
# ax[1].set_title("Pre-emphasized Spectrogram")
# plt.colorbar(img_1, format='%+2.0f dB')
# plt.colorbar(img_2, format='%+2.0f dB')
# plt.show()

# 梅尔滤波器组
# frame = 25 # 帧长
# hop_length = 10 # 帧移
# win_length = int(frame * sampling_rate_2 / 1000)
# hop_length = int(hop_length * sampling_rate_2 / 1000)
# n_fft = int(2 ** np.ceil(np.log2(win_length)))
# n_mels = 128
# mel_basis = librosa.filters.mel(sr=sampling_rate_2, n_fft=n_fft, n_mels=n_mels)
# print(f"mel_basis shape: {mel_basis.shape}") # 梅尔滤波器组矩阵的形状，通常为(n_mels, n_fft//2 + 1)
# x = np.arange(mel_basis.shape[1]) * sampling_rate_2 / n_fft
# plt.plot(x, mel_basis.T)
# plt.title("Mel Filter Bank")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Mel Filter Coefficients")
# plt.show()

# Fbank特征
# frame = 25 # 帧长
# hop_length = 10 # 帧移
# win_length = int(frame * sampling_rate_2 / 1000)
# hop_length = int(hop_length * sampling_rate_2 / 1000)
# n_fft = int(2 ** np.ceil(np.log2(win_length)))
# n_mels = 128
# fig = plt.figure()
# fbank = librosa.feature.melspectrogram(y=array_2, sr=sampling_rate_2, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=n_mels)
# print(f"f_bank shape: {fbank.shape}") # (n_mels, n_frames)
# fbank_dB = librosa.power_to_db(fbank, ref=np.max)
# img = librosa.display.specshow(fbank_dB, sr=sampling_rate_2, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='hot', fmax=sampling_rate_2/2)
# fig.colorbar(img, format='%+2.0f dB')
# plt.title("Mel—Frequency Spectrogram")
# plt.show()

# MFCC特征
# frame = 25 # 帧长
# hop_length = 10 # 帧移
# win_length = int(frame * sampling_rate_2 / 1000)
# hop_length = int(hop_length * sampling_rate_2 / 1000)
# n_fft = int(2 ** np.ceil(np.log2(win_length)))
# n_mels = 128
# n_mfcc = 20
# mfcc_1 = librosa.feature.mfcc(y=array_2, sr=sampling_rate_2, n_mfcc=n_mfcc, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length, win_length=win_length, dct_type=1)
# mfcc_2 = librosa.feature.mfcc(y=array_2, sr=sampling_rate_2, n_mfcc=n_mfcc, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length, win_length=win_length, dct_type=2)
# mfcc_3 = librosa.feature.mfcc(y=array_2, sr=sampling_rate_2, n_mfcc=n_mfcc, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length, win_length=win_length, dct_type=3)
# fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, constrained_layout=True)
# img_1 = librosa.display.specshow(mfcc_1, x_axis='time', ax=ax[0])
# img_2 = librosa.display.specshow(mfcc_2, x_axis='time', ax=ax[1])
# img_3 = librosa.display.specshow(mfcc_3, x_axis='time', ax=ax[2])
# ax[0].set_title("MFCC with DCT Type 1")
# ax[1].set_title("MFCC with DCT Type 2")
# ax[2].set_title("MFCC with DCT Type 3")
# fig.colorbar(img_1, ax=ax[0])
# fig.colorbar(img_2, ax=ax[1])
# fig.colorbar(img_3, ax=ax[2])
# plt.title('MFCC')
# plt.show()

# 特征拼接与差分
# frame = 25 # 帧长
# hop_length = 10 # 帧移
# win_length = int(frame * sampling_rate_2 / 1000)
# hop_length = int(hop_length * sampling_rate_2 / 1000)
# n_fft = int(2 ** np.ceil(np.log2(win_length)))
# n_mels = 128
# n_mfcc = 20
# mfcc = librosa.feature.mfcc(y=array_2, sr=sampling_rate_2, n_mfcc=n_mfcc, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length, win_length=win_length)
# mfcc_delta = librosa.feature.delta(mfcc)
# mfcc_delta_delta = librosa.feature.delta(mfcc, order=2)
# print(f"mfcc shape: {mfcc.shape}")
# print(f"mfcc_delta shape: {mfcc_delta.shape}")
# print(f"mfcc_delta_delta shape: {mfcc_delta_delta.shape}")
# mfcc_d1_d2 = np.concatenate((mfcc, mfcc_delta, mfcc_delta_delta), axis=0)
# fig = plt.figure()
# img = librosa.display.specshow(mfcc_d1_d2, x_axis='time')
# fig.colorbar(img)
# plt.title('MFCC with Delta and Delta-Delta')
# plt.show()

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