import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import librosa
import librosa.display

class ShipAudioFeatureExtractor:
    """
    船舶音频特征提取器
    """
    def __init__(self, sample_rate=None):
        """
        初始化特征提取器
        
        Args:
            sample_rate: 目标采样率，如果为None则保持原采样率
        """
        self.sample_rate = sample_rate
        
    def load_audio(self, file_path):
        """
        加载音频文件
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            waveform: 音频波形数据
            sr: 采样率
        """
        waveform, sr = torchaudio.load(file_path)
        
        # 如果是双声道，转换为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 重采样（如果需要）
        if self.sample_rate is not None and sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            sr = self.sample_rate
            
        return waveform, sr
    
    def extract_stft(self, waveform, sr, n_fft=2048, hop_length=512, win_length=None):
        """
        提取短时傅里叶变换特征
        
        Args:
            waveform: 音频波形数据
            sr: 采样率
            n_fft: FFT窗口大小
            hop_length: 帧移
            win_length: 窗口长度，默认等于n_fft
            
        Returns:
            spectrogram: 频谱图
            phase: 相位信息
        """
        if win_length is None:
            win_length = n_fft
        
        # 使用PyTorch的STFT
        stft = torch.stft(
            waveform[0],
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length).to(waveform.device),
            return_complex=True
        )
        
        # 计算幅度谱和相位谱
        spectrogram = torch.abs(stft)
        phase = torch.angle(stft)
        
        return spectrogram, phase
    
    def extract_spectrogram(self, waveform, sr, n_fft=2048, hop_length=512, win_length=None):
        """
        提取频谱图特征
        
        Args:
            waveform: 音频波形数据
            sr: 采样率
            n_fft: FFT窗口大小
            hop_length: 帧移
            win_length: 窗口长度，默认等于n_fft
            
        Returns:
            spectrogram_db: 分贝单位的频谱图
        """
        spectrogram, _ = self.extract_stft(waveform, sr, n_fft, hop_length, win_length)
        
        # 转换为分贝单位
        spectrogram_db = torchaudio.transforms.AmplitudeToDB()(spectrogram)
        
        return spectrogram_db
    
    def extract_mel_spectrogram(self, waveform, sr, n_fft=2048, hop_length=512, n_mels=128):
        """
        提取梅尔频谱图特征
        
        Args:
            waveform: 音频波形数据
            sr: 采样率
            n_fft: FFT窗口大小
            hop_length: 帧移
            n_mels: 梅尔滤波器组数量
            
        Returns:
            mel_spectrogram_db: 分贝单位的梅尔频谱图
        """
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )(waveform)
        
        # 转换为分贝单位
        mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
        
        return mel_spectrogram_db
    
    def extract_mfcc(self, waveform, sr, n_mfcc=13, n_fft=2048, hop_length=512, n_mels=128):
        """
        提取MFCC特征
        
        Args:
            waveform: 音频波形数据
            sr: 采样率
            n_mfcc: MFCC系数数量
            n_fft: FFT窗口大小
            hop_length: 帧移
            n_mels: 梅尔滤波器组数量
            
        Returns:
            mfcc: MFCC特征
        """
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'n_mels': n_mels
            }
        )
        
        mfcc = mfcc_transform(waveform)
        return mfcc
    
    def visualize_waveform(self, waveform, sr, title="音频波形"):
        """
        可视化音频波形
        """
        plt.figure(figsize=(12, 4))
        # 设置支持中文显示的字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        
        time_axis = torch.arange(0, waveform.shape[1]) / sr
        plt.plot(time_axis, waveform[0].numpy())
        plt.title(title)
        plt.xlabel("时间 (秒)")
        plt.ylabel("振幅")
        plt.tight_layout()
        plt.show()
    
    def visualize_spectrogram(self, spectrogram, sr, hop_length=512, title="频谱图"):
        """
        可视化频谱图
        """
        plt.figure(figsize=(12, 6))
        # 设置支持中文显示的字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        
        # 转换为numpy数组
        if torch.is_tensor(spectrogram):
            spectrogram_np = spectrogram.numpy()
        else:
            spectrogram_np = spectrogram
        
        # 确保数据是2D的
        if spectrogram_np.ndim == 1:
            spectrogram_np = spectrogram_np.reshape(1, -1)
        
        # 使用librosa进行可视化
        librosa.display.specshow(
            spectrogram_np,
            sr=sr,
            hop_length=hop_length,
            x_axis="time",
            y_axis="linear",
            cmap="viridis"
        )
        
        plt.colorbar(format="%+2.0f dB")
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def visualize_mel_spectrogram(self, mel_spectrogram, sr, hop_length=512, title="梅尔频谱图"):
        """
        可视化梅尔频谱图
        """
        plt.figure(figsize=(12, 6))
        # 设置支持中文显示的字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        
        # 转换为numpy数组
        if torch.is_tensor(mel_spectrogram):
            mel_spectrogram_np = mel_spectrogram.numpy()
        else:
            mel_spectrogram_np = mel_spectrogram
        
        # 确保数据是2D的
        if mel_spectrogram_np.ndim == 1:
            mel_spectrogram_np = mel_spectrogram_np.reshape(1, -1)
        
        # 使用librosa进行可视化
        librosa.display.specshow(
            mel_spectrogram_np,
            sr=sr,
            hop_length=hop_length,
            x_axis="time",
            y_axis="mel",
            cmap="viridis"
        )
        
        plt.colorbar(format="%+2.0f dB")
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def visualize_mfcc(self, mfcc, sr, hop_length=512, title="MFCC"):
        """
        可视化MFCC特征
        """
        plt.figure(figsize=(12, 6))
        # 设置支持中文显示的字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        
        # 转换为numpy数组
        if torch.is_tensor(mfcc):
            mfcc_np = mfcc.numpy()
        else:
            mfcc_np = mfcc
        
        # 确保数据是2D的
        if mfcc_np.ndim == 1:
            mfcc_np = mfcc_np.reshape(1, -1)
        
        # 使用librosa进行可视化
        librosa.display.specshow(
            mfcc_np,
            sr=sr,
            hop_length=hop_length,
            x_axis="time",
            cmap="viridis"
        )
        
        plt.colorbar(format="%+2.0f")
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def compute_and_visualize_cwt(self, waveform, sr, title="连续小波变换"):
        """
        计算并可视化连续小波变换 (CWT)
        """
        # 设置支持中文显示的字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        
        # 转换为numpy数组
        if torch.is_tensor(waveform):
            signal = waveform[0].numpy()
        else:
            signal = waveform
        
        # 计算奈奎斯特频率（采样率的一半）
        nyquist_freq = sr / 2
        
        # 自适应设置参数
        fmin = librosa.note_to_hz('C2')  # 起始频率
        
        # 计算最大可能的n_bins，确保最高频率不超过奈奎斯特频率
        # 每个八度音阶有12个音符，每增加12个bins，频率翻倍
        # 因此我们可以计算出在不超过奈奎斯特频率的情况下，最多可以有多少个bins
        max_octaves = np.floor(np.log2(nyquist_freq / fmin))
        max_bins = int(max_octaves * 12)
        
        # 使用计算出的最大bins数，但不超过84（为了可视化效果）
        n_bins = min(84, max_bins)
        
        print(f"使用参数: fmin={fmin:.2f} Hz, n_bins={n_bins}, 最大八度数={max_octaves:.2f}")
        
        # 使用librosa计算CQT (作为CWT的替代)
        try:
            C = librosa.cqt(
                signal, 
                sr=sr, 
                hop_length=512, 
                bins_per_octave=12, 
                n_bins=n_bins, 
                fmin=fmin
            )
            
            # 检查最高频率
            freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=12)
            max_freq = max(freqs)
            print(f"CQT最高频率: {max_freq:.2f} Hz, 奈奎斯特频率: {nyquist_freq:.2f} Hz")
            
            C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
            
            plt.figure(figsize=(12, 6))
            librosa.display.specshow(C_db, sr=sr, x_axis='time', y_axis='cqt_hz', cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
            plt.title(title)
            plt.tight_layout()
            plt.show()
    
    def analyze_ship_noise(self, file_path):
        """
        分析船舶辐射噪声并可视化时频特性
        
        Args:
            file_path: 音频文件路径
        """
        print(f"正在分析文件: {file_path}")
        
        # 加载音频
        waveform, sr = self.load_audio(file_path)
        print(f"音频信息: 采样率={sr}Hz, 时长={waveform.shape[1]/sr:.2f}秒")
        
        # 可视化波形
        self.visualize_waveform(waveform, sr, title="船舶辐射噪声波形")
        
        # 计算并可视化STFT频谱图
        spectrogram_db = self.extract_spectrogram(waveform, sr)

        if spectrogram_db.dim() > 2:
            self.visualize_spectrogram(spectrogram_db[0], sr, title="船舶辐射噪声STFT频谱图")
        else:
            self.visualize_spectrogram(spectrogram_db, sr, title="船舶辐射噪声STFT频谱图")
        
        # 计算并可视化梅尔频谱图
        mel_spectrogram_db = self.extract_mel_spectrogram(waveform, sr)

        if mel_spectrogram_db.dim() > 2:
            self.visualize_mel_spectrogram(mel_spectrogram_db[0], sr, title="船舶辐射噪声梅尔频谱图")
        else:
            self.visualize_mel_spectrogram(mel_spectrogram_db, sr, title="船舶辐射噪声梅尔频谱图")
        
        # 计算并可视化MFCC
        mfcc = self.extract_mfcc(waveform, sr)

        if mfcc.dim() > 2:
            self.visualize_mfcc(mfcc[0], sr, title="船舶辐射噪声MFCC")
        else:
            self.visualize_mfcc(mfcc, sr, title="船舶辐射噪声MFCC")
        
        # 计算并可视化连续小波变换
        self.compute_and_visualize_cwt(waveform, sr, title="船舶辐射噪声连续小波变换")
        
        print("分析完成!")
        
        return {
            "waveform": waveform,
            "sr": sr,
            "spectrogram": spectrogram_db,
            "mel_spectrogram": mel_spectrogram_db,
            "mfcc": mfcc
        }


# 示例用法
if __name__ == "__main__":
    # 创建特征提取器
    extractor = ShipAudioFeatureExtractor()
    
    # 替换为你的船舶噪声音频文件路径
    audio_file = r"E:\数据集\ShipEar\shipsEar_AUDIOS\6__10_07_13_marDeCangas_Entra.wav"
    
    # 如果文件存在则分析
    if Path(audio_file).exists():
        features = extractor.analyze_ship_noise(audio_file)
    else:
        print(f"文件不存在: {audio_file}")
        print("请提供有效的船舶噪声音频文件路径")