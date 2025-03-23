import os
import argparse
import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt
import torchaudio
from scipy import signal
from tqdm import tqdm

def extract_demon_librosa(audio_path, low_cut, high_cut, sr, n_fft, hop_length):
    """使用librosa提取DEMON特征"""
    # 加载音频文件
    y, sr = librosa.load(audio_path, sr=sr)
    
    # 带通滤波
    sos = signal.butter(N=4, Wn=[low_cut, high_cut], btype='bandpass', fs=sr, output='sos')
    filtered = signal.sosfilt(sos, y)
    
    # 计算包络（希尔伯特变换）
    analytic_signal = signal.hilbert(filtered)
    amplitude_envelope = np.abs(analytic_signal)
    
    # 计算DEMON谱
    stft = librosa.stft(amplitude_envelope, n_fft=n_fft, hop_length=hop_length)
    power_spectrum = np.abs(stft) ** 2
    
    # 归一化
    max_val = np.max(power_spectrum)
    demon_feature = power_spectrum / max_val if max_val > 0 else power_spectrum
    
    return demon_feature, sr

def extract_demon_pytorch(audio_path, low_cut, high_cut, sr, n_fft, hop_length, device):
    """使用PyTorch提取DEMON特征"""
    # 打印GPU信息
    if device == 'cuda' and torch.cuda.is_available():
        print(f"\n使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU处理")
    
    # 加载音频文件
    y, sr = librosa.load(audio_path, sr=sr)
    
    # 带通滤波 (使用scipy，因为PyTorch没有直接的滤波器实现)
    sos = signal.butter(N=4, Wn=[low_cut, high_cut], btype='bandpass', fs=sr, output='sos')
    filtered = signal.sosfilt(sos, y)
    
    # 计算包络（希尔伯特变换）
    analytic_signal = signal.hilbert(filtered)
    amplitude_envelope = np.abs(analytic_signal)
    
    # 转换为PyTorch张量
    envelope_tensor = torch.FloatTensor(amplitude_envelope).unsqueeze(0).to(device)
    
    # 创建STFT转换器
    stft_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2  # 平方，得到功率谱
    ).to(device)
    
    # 提取功率谱
    power_spectrum = stft_transform(envelope_tensor)
    
    # 归一化 - DEMON特征
    max_val = torch.max(power_spectrum)
    demon_feature = power_spectrum / max_val if max_val > 0 else power_spectrum
    
    # 转回CPU并转为numpy数组
    demon_feature = demon_feature.squeeze().cpu().numpy()
    
    return demon_feature, sr

def plot_demon(demon_feature, sr, hop_length, output_path):
    """绘制并保存DEMON特征图像"""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        librosa.amplitude_to_db(demon_feature, ref=np.max),
        sr=sr, 
        hop_length=hop_length,
        x_axis='time',
        y_axis='linear',
        vmin=-80,
        vmax=0
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('DEMON Feature')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_dataset(args):
    """处理整个数据集"""
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取所有音频文件
    audio_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(('.wav')):
                audio_files.append(os.path.join(root, file))
    
    # 使用tqdm显示进度
    for audio_path in tqdm(audio_files, desc=f"使用{args.method}提取DEMON特征"):
        # 创建相对路径结构
        rel_path = os.path.relpath(audio_path, args.input_dir) # 相对路径
        base_name = os.path.splitext(rel_path)[0] # 去掉扩展名
        output_path = os.path.join(args.output_dir, f"{base_name}.png") # 输出路径
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 提取DEMON特征
        if args.method == 'librosa':
            demon_feature, sr = extract_demon_librosa(
                audio_path, args.low_cut, args.high_cut, args.sr, args.n_fft, args.hop_length
            )
        elif args.method == 'pytorch':
            demon_feature, sr = extract_demon_pytorch(
                audio_path, args.low_cut, args.high_cut, args.sr, args.n_fft, args.hop_length, args.device
            )
        
        # 保存DEMON特征图像
        plot_demon(demon_feature, args.sr, args.hop_length, output_path)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='从ShipEar数据集提取DEMON特征')
    parser.add_argument('--input_dir', type=str, default=r'E:\数据集\ShipEar\shipsEar_AUDIOS', help='ShipEar数据集的输入目录')
    parser.add_argument('--output_dir', type=str, default=r'E:\数据集\ShipEar\Visual_Features\demon_features', help='DEMON特征输出目录')
    parser.add_argument('--method', type=str, choices=['librosa', 'pytorch'], default='pytorch', 
                        help='提取DEMON特征的方法 (librosa 或 pytorch)')
    parser.add_argument('--n_fft', type=int, default=2048, help='FFT窗口大小')
    parser.add_argument('--hop_length', type=int, default=512, help='帧移')
    parser.add_argument('--sr', type=int, default=44100, help='目标采样率')
    parser.add_argument('--low_cut', type=float, default=300.0, help='带通滤波器低截止频率 (Hz)')
    parser.add_argument('--high_cut', type=float, default=5000.0, help='带通滤波器高截止频率 (Hz)')
    parser.add_argument('--device', type=str, default='cuda', help='使用的设备 (cuda 或 cpu)')
    
    args = parser.parse_args()

    process_dataset(args)