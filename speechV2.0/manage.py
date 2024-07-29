#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
import time
from Speaker_Recognition import register, speakerrecog  # 声纹识别库

app = Flask(__name__)


# 首页面
@app.route("/")
def index():
    return render_template("index.html")


# 声纹注册
@app.route("/speech", methods=['GET', 'POST'])
def beginRecorder():
    printName = request.form.get('printNames')
    if not printName:
        return jsonify({"error": "未提供名字"}), 400

    try:
        begin = time.time()
        register.train_model(printName)
        duration = time.time() - begin
        return jsonify({"status": "成功", "duration": duration}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 结束录音
@app.route("/stopSpeech", methods=["GET", "POST"])
def stopRecorder():
    try:
        print("停止录音……")
        # speechRecorder.stop()  # 实现speechRecorder逻辑后取消注释
        return jsonify({"status": "录音已停止"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 说话人识别
@app.route("/recognize", methods=['GET', 'POST'])
def recognize():
    try:
        result = speakerrecog.speakerRecog()
        return jsonify({"result": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # 启动多线程参数，加快资源请求，快速响应用户
    app.run(debug=True, host='0.0.0.0', port=8600, threaded=True)
