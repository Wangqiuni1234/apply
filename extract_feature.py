#提取特征
import os.path

import numpy as np
from tqdm import tqdm
import librosa.feature


def extract_features(data_dir,save_dir,sr=16000,n_mfcc=13,n_mels=128):
    """提取特征并且保存"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for sub_folder in os.listdir(data_dir):
        sub_path = os.path.join(data_dir,sub_folder)
        if not os.path.isdir(sub_path):
            continue
        for emotion_folder in os.listdir(sub_path):
            emotion_path = os.path.join(sub_path,emotion_folder)
            if not os.path.isdir(emotion_path):
                continue
            save_path = os.path.join(save_dir,emotion_folder) # 创建保存特征的目录
            os.makedirs(save_path,exist_ok=True)

            for audio_file in tqdm(os.listdir(emotion_path),desc=f"Extracting features: {emotion_folder}"):
                file_path = os.path.join(emotion_path,audio_file)
                if not file_path.endswith(".wav"):
                    continue
                audio,sr = librosa.load(file_path,sr=sr) # 加载音频文件
                stft = np.abs(librosa.stft(audio))
                mfcc = librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=n_mfcc) # 提取MFCC相关特征
                mfcc_del1 = librosa.feature.delta(mfcc)
                mfcc_del2 = librosa.feature.delta(mfcc,order=2)
                mel_spectrogram = librosa.feature.melspectrogram(y=audio,sr=sr,n_mels=n_mels) # 获取梅尔语谱图特征
                mfcc_mean = np.mean(mfcc,axis=1)
                mfcc_var = np.var(mfcc,axis=1)
                mfcc_del1_mean = np.mean(mfcc_del1,axis=1)
                mfcc_del1_var = np.var(mfcc_del1,axis=1)
                mfcc_del2_mean = np.mean(mfcc_del2,axis=1)
                mfcc_del2_var = np.var(mfcc_del2, axis=1)
                mel_spectrogram_mean = np.mean(mel_spectrogram,axis=1)
                mel_spectrogram_var = np.var(mel_spectrogram,axis=1)
                # 色谱图
                chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
                # 均方根能量
                S, phase = librosa.magphase(stft)
                meanMagnitude = np.mean(S)
                rmse = librosa.feature.rms(S=S)[0]
                meanrms = np.mean(rmse)
                # 过零率
                zerocr = np.mean(librosa.feature.zero_crossing_rate(audio))
                # 使用YIN算法估计基频（更鲁棒、不同情绪波动大）
                pitch = librosa.yin(audio, fmin=80, fmax=400)
                pitch = pitch[~np.isnan(pitch)]  # 去除NaN值
                pitch_mean = np.mean(pitch) if len(pitch) > 0 else 0.0
                pitch_var = np.var(pitch) if len(pitch) > 0 else 0.0
                pitch_flatness = np.mean(np.abs(np.diff(pitch)))  # 相邻基频差均值、基频平坦度（Pitch Flatness）
                # 频谱质心（反应声音的明亮度、生气或快乐时更高）
                spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
                sc_mean = np.mean(spectral_centroid)
                sc_var = np.var(spectral_centroid)
                # 频谱带宽（描述频率分布范围，生气可能带宽更大）
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
                sb_mean = np.mean(spectral_bandwidth)
                sb_var = np.var(spectral_bandwidth)
                # 频谱平坦度（neutral更平坦）
                spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
                sf_mean = np.mean(spectral_flatness)
                featuret = np.array([
                     meanMagnitude,meanrms,zerocr
                ])
                features = np.concatenate([featuret,chroma, mfcc_mean,mfcc_var,mfcc_del1_mean,mfcc_del1_var,mfcc_del2_mean,mfcc_del2_var,mel_spectrogram_mean,mel_spectrogram_var],axis=0)
                # print(f"features的形状： {features.shape}")
                # 保存特征到文件
                feature_path = os.path.join(save_path,f"{os.path.splitext(audio_file)[0]}_feature.npy")
                np.save(feature_path,features)
                # print(f"saved features for {audio_file} at {feature_path}")


if __name__ == "__main__":
    data_dir = "dataset/"
    save_dir = "savedir/saved_features/"
    extract_features(data_dir,save_dir)