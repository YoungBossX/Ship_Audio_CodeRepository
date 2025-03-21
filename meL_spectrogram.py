import os
import argparse
import numpy as np
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torchaudio
from tqdm import tqdm

def extract_mel_spectrogram_librosa(audio_path, n_fft, hop_length, n_mels, sr):
    """使用librosa提取梅尔谱特征"""
    # 加载音频文件
    y, sr = librosa.load(audio_path, sr=sr)
    
    # 提取梅尔谱
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    
    # 不再转换为分贝单位，直接返回原始梅尔谱
    return mel_spectrogram, sr

def extract_mel_spectrogram_pytorch(audio_path, n_fft, hop_length, n_mels, sr, device):
    """使用PyTorch提取梅尔谱特征"""
    # 打印GPU信息
    if device == 'cuda' and torch.cuda.is_available():
        print(f"\n使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU处理")
    
    # 加载音频文件
    y, sr = librosa.load(audio_path, sr=sr)
    y_tensor = torch.FloatTensor(y).unsqueeze(0).to(device)  # 添加批次维度
    
    # 创建梅尔谱转换器
    mel_transform = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
    ).to(device)
    
    # 提取梅尔谱
    mel_spectrogram = mel_transform(y_tensor)
    
    # 转回CPU并转为numpy数组
    mel_spectrogram = mel_spectrogram.squeeze().cpu().numpy()
    
    return mel_spectrogram, sr

def plot_mel_spectrogram(mel_spectrogram, sr, hop_length, output_path):
    """绘制并保存梅尔谱图像"""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_spectrogram, 
        sr=sr, 
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel'
    )
    plt.colorbar(format='%+2.0f')  # 修改格式，不再使用dB单位
    plt.title('Mel Spectrogram (Raw)')  # 修改标题，表明是原始梅尔谱
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_dataset(input_dir, output_dir, method, n_fft, hop_length, n_mels, sr, device):
    """处理整个数据集"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有音频文件
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.wav')):
                audio_files.append(os.path.join(root, file))
    
    # 使用tqdm显示进度
    for audio_path in tqdm(audio_files, desc=f"使用{method}提取原始梅尔谱"):
        # 创建相对路径结构
        rel_path = os.path.relpath(audio_path, input_dir)
        base_name = os.path.splitext(rel_path)[0]
        output_path = os.path.join(output_dir, f"{base_name}.png")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 提取梅尔谱
        if method == 'librosa':
            mel_spectrogram, sr = extract_mel_spectrogram_librosa(
                audio_path, n_fft, hop_length, n_mels, sr
            )
        elif method == 'pytorch': # pytorch
            mel_spectrogram, sr = extract_mel_spectrogram_pytorch(
                audio_path, n_fft, hop_length, n_mels, sr, device
            )
        
        # 保存梅尔谱图像
        plot_mel_spectrogram(mel_spectrogram, sr, hop_length, output_path)
        
        # 可选：保存梅尔谱数据
        np_output_path = os.path.join(output_dir, f"{base_name}.npy")
        np.save(np_output_path, mel_spectrogram)

def main():
    parser = argparse.ArgumentParser(description='从ShipEar数据集提取原始梅尔谱特征')
    parser.add_argument('--input_dir', type=str, required=True, help='ShipEar数据集的输入目录')
    parser.add_argument('--output_dir', type=str, required=True, help='梅尔谱输出目录')
    parser.add_argument('--method', type=str, choices=['librosa', 'pytorch'], default='librosa', 
                        help='提取梅尔谱的方法 (librosa 或 pytorch)')
    parser.add_argument('--n_fft', type=int, default=2048, help='FFT窗口大小')
    parser.add_argument('--hop_length', type=int, default=512, help='帧移')
    parser.add_argument('--n_mels', type=int, default=128, help='梅尔滤波器组数量')
    parser.add_argument('--sr', type=int, default=44100, help='目标采样率，None表示使用原始采样率')
    parser.add_argument('--device', type=str, default='cuda', help='使用的设备 (cuda 或 cpu)')
    
    # 提供默认参数列表
    args = parser.parse_args([
        '--input_dir', 'D:\数据集\shipsEar_AUDIOS',
        '--output_dir', 'raw_mel_spectrum',  # 修改输出目录名称
        '--method', 'pytorch',
        '--n_fft', '2048',
        '--hop_length', '512',
        '--n_mels', '128',
        '--sr', '44100',
        '--device', 'cuda'
    ])
    
    process_dataset(
        args.input_dir, 
        args.output_dir, 
        args.method,
        args.n_fft,
        args.hop_length,
        args.n_mels,
        args.sr,
        args.device
    )

if __name__ == "__main__":
    main()