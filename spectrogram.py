import os
import argparse
import torch
import torchaudio
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='船舶辐射噪声频谱图特征提取')
    parser.add_argument('--input_dir', type=str, default="D:\数据集\shipsEar_AUDIOS", help='输入音频文件目录')
    parser.add_argument('--output_dir', type=str, default="D:\数据集\特征可视化\ShipEar", help='输出特征保存目录')
    parser.add_argument('--sample_rate', type=int, default=44100, help='采样率')
    parser.add_argument('--n_fft', type=int, default=1024, help='FFT大小')
    parser.add_argument('--hop_length', type=int, default=512, help='帧移')
    parser.add_argument('--save_plots', action='store_false', help='是否保存频谱图可视化结果')
    return parser.parse_args()

def extract_features(audio_path, args, device):
    """提取音频特征"""
    # 加载音频并立即移至指定设备
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.to(device)
    
    # 重采样（如果需要）
    if sample_rate != args.sample_rate:
        resampler = torchaudio.transforms.Resample(
            sample_rate, 
            args.sample_rate
        ).to(device)
        waveform = resampler(waveform)
    
    # 提取STFT频谱图
    spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        power=2.0
    ).to(device)
    
    spec = spectrogram_transform(waveform)
    
    # 转换为分贝单位（确保在同一设备上）
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(device)
    spec_db = amplitude_to_db(spec)
    
    # 将张量移回CPU并转换为NumPy数组
    spec_db = spec_db.cpu().numpy()
    
    return {
        'spectrogram': spec_db,
        'file_name': os.path.basename(audio_path)
    }

def save_features(features, output_path, save_plots):
    """保存提取的特征
    
    Args:
        features: 特征字典
        output_path: 输出路径
        save_plots: 是否保存可视化结果
    """
    # 可视化并保存图像
    if save_plots:
        # 频谱图
        plt.figure(figsize=(10, 4))
        plt.imshow(features['spectrogram'][0], aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram - {features["file_name"]}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"{features['file_name']}_spec.png"))
        plt.close()

def main():
    """主函数"""
    args = parse_args()
    
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取所有音频文件
    audio_files = []
    for ext in ['.wav']:
        audio_files.extend(list(Path(args.input_dir).glob(f'**/*{ext}')))
    
    print(f"找到 {len(audio_files)} 个音频文件")
    
    # 处理每个音频文件
    for audio_path in tqdm(audio_files, desc="处理音频文件"):
        try:
            # 提取特征
            features = extract_features(str(audio_path), args, device)
            
            # 保存特征
            save_features(features, args.output_dir, args.save_plots)
            
        except Exception as e:
            print(f"处理文件 {audio_path} 时出错: {e}")
    
    print("特征提取完成！")

if __name__ == "__main__":
    main()