import tkinter as tk
from tkinter import filedialog, messagebox
import soundfile as sf
import sounddevice as sd
import os
from predict import *
from models.mymodel import CombineModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AudioApp:
    def __init__(self, master):
        self.master = master
        self.master.title("语音情感识别操作界面")
        self.master.geometry("500x500")

        # 加载训练好的模型
        self.models = CombineModel(num_class=4).to(device)
        # self.models.load_state_dict(torch.load('/Users/didi/Desktop/fournum_class/results2000/my_new_best_model_1000.pth',
        # map_location=torch.device('cpu')))# 加载训练好的模型
        self.models.load_state_dict(
            torch.load('D:\\graduationProject\\apply\\results2000\\my_new_best_model_1000.pth',
                       map_location=torch.device('cpu')))  # 加载训练好的模型
        self.models.eval()  # 设置为评估模式

        self.audio_path = None # 当前音频文件路径
        self.file_label = tk.Label(master,text="当前音频文件：无",wraplength=300)
        self.file_label.pack(pady=10)

        self.select_button = tk.Button(master,text="1.从文件选择",command=self.select_file)
        self.select_button.pack(padx=30, pady=10)
        self.record_button = tk.Button(master,text="2.开始录音",command=self.record_audio)
        self.record_button.pack(padx=30,pady=10)

        self.play_button = tk.Button(master,text="播放音频",command=self.play_audio,state=tk.DISABLED)
        self.play_button.pack(pady=10)

        self.predict_button = tk.Button(master,text="预测此音频的情感",command=self.to_predict)
        self.predict_button.pack(pady=10)

        self.result_label = tk.Label(master, text="当前音频情感类别：", wraplength=300)
        self.result_label.pack(pady=10)

    def select_file(self):
        # 从文件里面选择文件并更新存储区域
        file_path = filedialog.askopenfilename(filetypes=[("WAV文件","*.wav")])
        if not file_path:
            return
        self.audio_path = file_path
        self.update_file_label()
        self.enable_play_button()

    def record_audio(self):
        # 开始录音
        self.is_recording = True
        self.audio_data = [] # 清空之前的录音数据
        self.recording_stream = sd.InputStream(callback=self.record_callback,channels=1,dtype='float32',samplerate=16000)
        self.recording_stream.start()

        self.recording_window = tk.Toplevel(self.master)
        self.recording_window.title("录音中")
        self.recording_window.geometry("300x100")
        tk.Label(self.recording_window,text="正在录音……").pack(pady=10)
        end_button = tk.Button(self.recording_window,text="结束录音",command=self.stop_recording)
        end_button.pack(pady=10)

        self.record_button.config(state=tk.DISABLED) # 禁用开始录音按钮

    def record_callback(self,indata,frames, time,status):
        # 录音回调函数
        if status:
            print(status)
        if self.is_recording:
            self.audio_data.append(indata.copy())

    def stop_recording(self):
        # 结束录音
        self.is_recording = False
        self.recording_stream.stop()

        # 检查是否有录音数据
        if not self.audio_data or len(self.audio_data) == 0:
            messagebox.showerror("错误", "没有录制到音频数据。")
            return

        audio = np.concatenate(self.audio_data,axis=0) # 将录音数据合并成一个numpy数组

        if not os.path.exists('audio/'):
            os.makedirs('audio/')  # 如果输出的目录不存在则创建
        file_path = "audio/record_audio.wav"  # 保存录音文件

        sf.write(file_path,audio,16000)
        self.audio_path = file_path

        self.update_file_label()
        messagebox.showinfo("录音完成",f"录音已完成并保存为 {file_path}")
        self.recording_window.destroy() # 关闭录音窗口
        self.enable_play_button()
        self.record_button.config(state=tk.NORMAL) # 启动开始录音按钮

    def play_audio(self):
        # 播放音频
        if not self.audio_path:
            messagebox.showerror("错误","没有音频文件可播放")
            return
        try:
            data,samplerate = sf.read(self.audio_path)
            sd.play(data,samplerate)
            sd.wait()
            messagebox.showinfo("播放完成","音频播放结束")
        except Exception as e:
            messagebox.showerror("错误",f"无法播放音频: {str(e)}")

    def update_file_label(self):
        # 更新文件标签
        if self.audio_path:
            self.file_label.config(text=f"当前音频文件： {self.audio_path}")
        else:
            self.file_label.config(text="当前音频文件： 无")

    def enable_play_button(self):
        # 启用播放按钮
        self.play_button.config(state=tk.NORMAL)

    def to_predict(self):
        if self.audio_path:
            # 清空之前的情感类别结果
            self.result_label.config(text="……")
            self.master.update()  # 强制刷新界面，确保"正在预测中"显示出来
            try:
                print(f"开始预测，使用的音频路径是: {self.audio_path}")  # 调试信息

                # 直接使用加载的训练好的模型进行预测
                predict_result = predict(self.audio_path, self.models)  # 使用现有模型进行预测
                en_emotion = ['angry', 'happy', 'neutral', 'sad']
                ch_resultindex = ['生气','开心','中性','伤心']
                ch_result = dict(zip(en_emotion,ch_resultindex))  # 映射
                ch_label = ch_result.get(predict_result)
                self.result_label.config(text=f"当前音频的情感类别为： {ch_label}")
            except Exception as e:
                self.result_label.config(text="预测失败，请重试")
                messagebox.showerror("错误", f"预测过程中发生错误：{str(e)}")
        else:
            messagebox.showerror("还没有选择音频文件哦！")


if __name__ == '__main__':
    root = tk.Tk()
    app = AudioApp(root)
    root.mainloop()
