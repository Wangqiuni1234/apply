import librosa

if __name__ == '__main__':
    try:
        audio, sr = librosa.load('audio/voice.wav', sr=16000)  # 显式指定采样率
        print("Audio shape:", audio.shape)
        print("Sample rate:", sr)
    except Exception as e:
        print("Error loading audio:", e)