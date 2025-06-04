#预处理
import os
import librosa
import numpy as np
from scipy.io.wavfile import write
from tqdm import tqdm


def load_audio(file_path,sr=16000):
    """加载音频文件"""
    audio,sr = librosa.load(file_path,sr=sr)
    return audio,sr

def apply_denoising(audio):
    """降噪"""
    stft = librosa.stft(audio)
    magnitude,phase = librosa.magphase(stft) # 幅度谱和相位谱

    noise_profile = np.median(magnitude,axis=1) # 计算噪声轮廓
    denoised_magnitude = np.maximum(magnitude - noise_profile[:,np.newaxis],0) # 谱减法

    denoised_audio = librosa.istft(denoised_magnitude * phase) # 进行逆stft以重建音频信号
    return denoised_audio

def wiener_filter(audio,sr,noise_magnitude=None):
    """维纳滤波对音频信号进行降噪"""
    stft = librosa.stft(audio)
    magnitude,phase = librosa.magphase(stft)
    if noise_magnitude is None:
        noise_magnitude = np.mean(magnitude[:,:10],axis=1) # 如果没有噪声的幅度谱，使用前几个帧估计噪声的幅度谱--10帧
    sigal_power = magnitude**2
    noise_power = noise_magnitude**2 # 功率谱
    wiener_filter = sigal_power / (sigal_power + noise_power) # 计算维纳滤波器的频率响应
    denoised_magnitude = magnitude * np.sqrt(wiener_filter)  # 应用维纳滤波到幅度谱上
    denoised_audio = denoised_magnitude * phase # 重建信号
    denoised_audio = librosa.istft(denoised_audio) # 使用逆stft重建时域信号
    return denoised_audio


def split_audio(audio,sr,sd=6):
    """分割音频"""
    segment_length = int(sr*sd) # 计算音频应该包含的样本数
    # segments = [audio[i:i+segment_length]
    #             for i in range(0,len(audio),segment_length)
    #                 if len(audio[i:i+segment_length]) == segment_length]
    segments = []

    for i in range(0, len(audio), segment_length):
        segment = audio[i:i + segment_length]
        if len(segment) < segment_length:  # 如果长度不足，进行补零
            padding = np.zeros(segment_length - len(segment), dtype=audio.dtype)
            segment = np.concatenate((segment, padding))
        segments.append(segment)
    return segments

# def augment_audio(audio,sr):
#     """数据增强---添加噪声、时间拉伸和时移"""
#     augmented_audios = []
#     noise = np.random.normal(0,0.005,len(audio))
#     augmented_audios.append(audio+noise) # 添加高斯噪声
#     stretched_audio = librosa.effects.time_stretch(audio,rate=0.8)
#     augmented_audios.append(stretched_audio[:len(audio)]) # 确保时间拉伸后的音频长度一致
#     shift = int(0.2*sr) # 计算时移样本数
#     shifted_audio = np.roll(audio,shift) # 时移
#     augmented_audios.append(shifted_audio)
#
#     return augmented_audios


# 当前增强方法可能引入过多噪声，且时移操作可能破坏情感连续性
# 改进方案：添加更合理的增强方式

def augment_audio(audio, sr):
    augmented_audios = []

    # 1. 速度扰动 (更温和的范围)
    rate = np.random.uniform(0.9, 1.1)
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    if len(stretched) < len(audio):
        stretched = np.pad(stretched, (0, max(0, len(audio) - len(stretched))))
    else:
        stretched = stretched[:len(audio)]
    augmented_audios.append(stretched)

    # 2. 音高偏移 (±2个半音)
    n_steps = np.random.randint(-2, 2)
    pitched = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    augmented_audios.append(pitched)

    # 3. 随机掩码 (模拟说话停顿)
    mask_len = int(0.1 * sr)  # 最大100ms
    start = np.random.randint(0, len(audio) - mask_len)
    masked = audio.copy()
    masked[start:start + mask_len] = 0
    augmented_audios.append(masked)

    return augmented_audios


def preprocess_dataset(input_dir,output_dir,sr=16000,sd=6):
    """对dataset数据集进行预处理，包括数据降噪、分割和数据增强

    args:
        input_dir-----原始数据集路径
        output_dir----预处理后的数据保存路径
        sr------------目标采样率
        sd------------分割的片段的时长
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) # 如果输出的目录不存在则创建

    for emotion_folder in os.listdir(input_dir):
        # 遍历
        emotion_path = os.path.join(input_dir,emotion_folder)
        if not os.path.isdir(emotion_path):
            continue # 跳过非目录文件
        # 创建输出目录中对应的情感类别文件
        output_emotion_path = os.path.join(output_dir,emotion_folder)
        os.makedirs(output_emotion_path,exist_ok=True)

        for audio_file in tqdm(os.listdir(emotion_path),desc=f"Preprocessing: {emotion_folder}"):
            file_path = os.path.join(emotion_path,audio_file)
            # 加载音频文件
            audio,sr = load_audio(file_path,sr)
            # 降噪
            audio = apply_denoising(audio)
            # 切割
            segments = split_audio(audio,sr,sd)
            # 数据增强
            all_segments = []
            for segment in segments:
                all_segments.append(segment)
                all_segments.extend(augment_audio(segment,sr))

            # 保存预处理后的音频片段到输出目录
            for i, segment in enumerate(all_segments):
                output_file = os.path.join(output_emotion_path,f"{os.path.splitext(audio_file)[0]}_{i}.wav")
                write(output_file,sr,(segment*32767).astype(np.int32))



def preprocess_audio(emotion_path,output_dir='audio/new_audio/',sr=16000,sd=6):
    """对单个音频文件进行预处理，包括数据清洗、分割和数据增强

    args:
        emotion_path-----原始数据集路径
        output_path-----处理后的文件路径
        sr------------目标采样率
        sd------------分割的片段的时长
    """
    # 加载音频文件
    audio,sr = load_audio(emotion_path,sr)
    # 降噪
    audio = apply_denoising(audio)
    # 切割
    segments = split_audio(audio,sr,sd)
    # 数据增强
    all_segments = []
    for segment in segments:
        # all_segments.append(segment)
        all_segments.extend(augment_audio(segment,sr))

        # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存预处理后的音频片段到输出目录
    for i, segment in enumerate(all_segments):
        # 根据片段生成动态文件名，避免覆盖
        output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(emotion_path))[0]}_{i}.wav")
        write(output_file,sr,(segment*32767).astype(np.int32))

    return output_file





if __name__ == "__main__":
    input_directory = "D:\\graduationProject\\apply\\dataset\\original"

    output_directory = "D:\\graduationProject\\apply\\dataset\\processed_dataset"

    preprocess_dataset(input_directory,output_directory)
