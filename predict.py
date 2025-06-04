# 对文件夹中的录音文件或者通过麦克风接收音频从而进行语音情感识别
import traceback

import torch
import librosa
import numpy as np
from models.mymodel import CombineModel
# from other.preprocess import preprocess_audio
from preprocess import preprocess_audio


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
target_names = ['angry','happy','neutral','sad']

models = CombineModel(num_class=4).to(device)
# models.load_state_dict(torch.load('models/best_model.pth'))
models.load_state_dict(
            torch.load('D:\\graduationProject\\apply\\results2000\\my_new_best_model_1000.pth',
                       map_location=torch.device('cpu')))  # 加载训练好的模型
models.eval()  # 设置为评估模式

def extract_feature_of_audio(audio_path,n_mfcc=13,n_mels=128):
    # try:
    #     audio,sr = librosa.load(audio_path,sr=None)

    try:
        audio, sr = librosa.load(audio_path, sr=None)
        print(f"成功加载音频文件，采样率：{sr}")
    except Exception as e:
        print(f"加载音频文件时发生错误：{e}")
        return None

        stft = np.abs(librosa.stft(audio))
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)  # 提取MFCC相关特征
        mfcc_del1 = librosa.feature.delta(mfcc)
        mfcc_del2 = librosa.feature.delta(mfcc, order=2)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)  # 获取梅尔语谱图特征
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_var = np.var(mfcc, axis=1)
        mfcc_del1_mean = np.mean(mfcc_del1, axis=1)
        mfcc_del1_var = np.var(mfcc_del1, axis=1)
        mfcc_del2_mean = np.mean(mfcc_del2, axis=1)
        mfcc_del2_var = np.var(mfcc_del2, axis=1)
        mel_spectrogram_mean = np.mean(mel_spectrogram, axis=1)
        mel_spectrogram_var = np.var(mel_spectrogram, axis=1)
        # 色谱图
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        # 均方根能量
        S, phase = librosa.magphase(stft)
        meanMagnitude = np.mean(S)
        rmse = librosa.feature.rms(S=S)[0]
        meanrms = np.mean(rmse)
        # 过零率
        zerocr = np.mean(librosa.feature.zero_crossing_rate(audio))
        featuret = np.array([
            meanMagnitude, meanrms, zerocr
        ])
        features = np.concatenate(
            [featuret, chroma, mfcc_mean, mfcc_var, mfcc_del1_mean, mfcc_del1_var, mfcc_del2_mean, mfcc_del2_var,
             mel_spectrogram_mean, mel_spectrogram_var], axis=0)

        return features

    except Exception as e:
        print(f"特征提取出错：{e}")
        traceback.print_exc()  # 打印详细错误栈
        return None

def predict(audio_path_f,models):
    audio_path = preprocess_audio(audio_path_f)

    if not audio_path.endswith('.wav'):
        raise ValueError("文件格式不支持，只支持 .wav 格式")

    features = extract_feature_of_audio(audio_path)
    if features is None:
        print("特征提取失败")
        return None
    features = torch.tensor(features,dtype=torch.float32).unsqueeze(0).to(device) # 将特征转换为张量并送入设备
    models.eval()
    with torch.no_grad():
        outputs = models(features)
        _,predicted = torch.max(outputs,1)
        print(f"识别到的情感：{target_names[predicted.item()]}")   #添加调试
        return target_names[predicted.item()]