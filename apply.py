from flask import Flask, request, jsonify
import os
from predict import *
from flask_cors import CORS
import traceback
from flask import send_from_directory


app = Flask(__name__)
CORS(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = CombineModel(num_class=4).to(device)
models.load_state_dict(
            torch.load('D:\\graduationProject\\apply\\results2000\\my_new_best_model_1000.pth',
                       map_location=torch.device('cpu')))  # 加载训练好的模型
models.eval()  # 设置为评估模式



# @app.route('/predict_emotion', methods=['POST'])
# def predict_emotion():
#
#     try:
#         if 'audio' not in request.files:
#             return jsonify({"error": "No file part"}), 400
#
#         file = request.files['audio']
#
#         if file.filename == '':
#             print("未选择文件")
#             return jsonify({"error": "No selected file"}), 400
#
#         # 保存临时文件并进行处理
#         # file_path = os.path.join('audio', file.filename)
#         file_path = os.path.join('audio', 'voice.wav')
#         file.save(file_path)
#
#         # # 确保转换后的路径是 .wav 格式
#         # converted_path = os.path.splitext(file_path)[0] + '.wav'
#         # predict_result = predict(converted_path, models)
#
#         predict_result = predict('audio/voice.wav', models)  # 使用现有模型进行预测
#         if not predict_result:
#             return jsonify({"error": "Emotion prediction failed"}), 500
#
#         # 如果识别到愤怒情感，返回"转人工客服"szx
#         if predict_result == 'angry':
#             return jsonify({"emotion": predict_result, "action": "transfer_to_human"})
#         else:
#             return jsonify({"emotion": predict_result, "action": "continue_listening"})
#
#     except Exception as e:
#         traceback.print_exc()  # 打印完整错误栈
#         return jsonify({"error": str(e)}), 500
#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)
#     # app.run(debug=True)
#
@app.route('/apply.html')
def serve_apply_html():
    return send_from_directory(directory='.', path='apply.html')  #


@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    try:
        if 'audio' not in request.files:
            print("没有收到音频文件")
            return jsonify({"error": "没有收到音频文件"}), 400

        file = request.files['audio']

        # 安全地获取文件名
        try:
            filename = file.filename if file.filename else "unnamed_file"
            print(f"Received file: {filename}")
        except Exception as e:
            print(f"获取文件名时出错: {str(e)}")
            filename = "unnamed_file"

        # 确保文件夹存在
        try:
            if not os.path.exists('audio'):
                os.makedirs('audio')
        except Exception as e:
            print(f"创建文件夹时出错: {str(e)}")
            return jsonify({"error": f"Failed to create directory: {str(e)}"}), 500

        # 保存文件，使用一个固定的名称
        file_path = os.path.join('audio', 'voice.wav')
        try:
            file.save(file_path)
            print(f"文件已保存到: {file_path}")
        except Exception as e:
            print(f"保存文件时出错: {str(e)}")
            return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

        # 临时跳过预测，仅返回一个硬编码结果
        # predict_result = predict(file_path, models)
        predict_result = "happy"  # 硬编码测试结果

        return jsonify({"emotion": predict_result, "action": "continue_listening"})

    except Exception as e:
        print(f"处理请求时出错: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    # app.run(debug=True)
