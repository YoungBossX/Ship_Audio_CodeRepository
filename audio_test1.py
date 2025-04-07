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

array, sampling_rate = librosa.load(r"D:\数据集\shipsEar_AUDIOS\7__10_07_13_marDeCangas_Espera.wav")
print(f"array: {array}")
print(f"array shape: {array.shape}") # (6644996,)
print(f"sampling_rate: {sampling_rate}") # 22050

# 波形图
plt.figure(figsize=(12, 8))
plt.ylabel("Amplitude")
librosa.display.waveshow(array, sr=sampling_rate)
plt.show()

# 频谱图
dft_input = array[:]
# 计算DFT
window = np.hanning(len(dft_input))
windowed_input = dft_input * window
dft = np.fft.rfft(windowed_input)
# 计算频谱的幅值
amplitude = np.abs(dft)
amplitude_db = librosa.amplitude_to_db(amplitude, ref=np.max)
# 计算每个DFT分量对应的频率值
frequency = librosa.fft_frequencies(sr=sampling_rate, n_fft=len(dft_input))
plt.figure().set_figwidth(12)
plt.plot(frequency, amplitude_db)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (dB)")
plt.xscale("log")
plt.show()

# 计算STFT，时频谱
D = librosa.stft(array)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
plt.figure().set_figwidth(12)
librosa.display.specshow(S_db, x_axis="time", y_axis="hz")
plt.colorbar()
plt.show()

# 梅尔谱图
S = librosa.feature.melspectrogram(y=array, sr=sampling_rate, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)
plt.figure().set_figwidth(12)
librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sampling_rate, fmax=8000)
plt.colorbar()
plt.show()