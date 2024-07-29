"""说话人识别"""
import pyaudio
import wave
import os
import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from .mfcc_coeff import extract_features
import warnings

warnings.filterwarnings("ignore")


def record_audio(output_filename, record_seconds=5, chunk=1024, format=pyaudio.paInt16, channels=2, rate=44100):
    """
    录制音频并保存为WAV文件。

    参数：
    - output_filename: 输出文件名
    - record_seconds: 录制时间（秒）
    - chunk: 音频块大小
    - format: 音频格式
    - channels: 通道数
    - rate: 采样率
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    print("* 录音中")
    frames = []

    for _ in range(0, int(rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    print("* 录音完成")

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))


def load_models(modelpath):
    """
    加载GMM模型。

    参数：
    - modelpath: 模型路径

    返回：
    - models: 加载的模型
    - speakers: 模型对应的说话人
    """
    gmm_files = [os.path.join(modelpath, fname) for fname in os.listdir(modelpath) if fname.endswith(".gmm")]
    models = [cPickle.load(open(fname, 'rb')) for fname in gmm_files]
    speakers = [os.path.splitext(os.path.basename(fname))[0] for fname in gmm_files]
    return models, speakers


def recognize_speaker(audio_path, models, speakers):
    """
    识别说话人。

    参数：
    - audio_path: 音频文件路径
    - models: 已加载的模型
    - speakers: 模型对应的说话人

    返回：
    - recognized_speaker: 识别出的说话人
    """
    sr, audio = read(audio_path)
    vector = extract_features(audio, sr)

    log_likelihood = np.zeros(len(models))

    for i, gmm in enumerate(models):
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()

    winner = np.argmax(log_likelihood)
    return speakers[winner]


def speakerRecog():
    """
    进行说话人识别并返回识别结果。
    """
    output_filename = ".\\Speaker_Recognition\\samples\\test.wav"
    modelpath = ".\\Speaker_Recognition\\gmm_models\\"

    # 录制音频
    record_audio(output_filename)

    # 加载模型
    models, speakers = load_models(modelpath)

    # 识别说话人
    recognized_speaker = recognize_speaker(output_filename, models, speakers)
    print("识别为 - ", recognized_speaker)

    return recognized_speaker
