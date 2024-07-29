"""声纹注册"""
# code=utf-8
import pyaudio
import wave
import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture
from .mfcc_coeff import extract_features
import warnings
import os

warnings.filterwarnings("ignore")


def recordVoice(word, speakerName, duration=4, chunk=1024, format=pyaudio.paInt16, channels=2, rate=44100):
    """
    录制特定单词的语音并保存为WAV文件。

    参数:
        word (str): 要录制的单词。
        speakerName (str): 说话者的名字。
        duration (int): 录音时长（秒）。默认值为4秒。
        chunk (int): 每个音频块的大小。默认值为1024。
        format (int): 音频格式。默认值为pyaudio.paInt16。
        channels (int): 音频通道数。默认值为2。
        rate (int): 采样率。默认值为44100 Hz。
    """
    output_dir = os.path.join(".\\Speaker_Recognition\\samples\\", "{}-2024".format(speakerName))
    os.makedirs(output_dir, exist_ok=True)
    WAVE_OUTPUT_FILENAME = os.path.join(output_dir, "{}_{}.wav".format(speakerName, word))

    p = pyaudio.PyAudio()
    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    print("正在录制 '{}'".format(word))

    frames = []

    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("* 录制完成")

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))


# return WAVE_OUTPUT_FILENAME

def train_model(speakerName):
    """
    为指定的说话者训练高斯混合模型（GMM）。

    参数:
        speakerName (str): 说话者的名字。
    """
    training_file_path = '.\\Speaker_Recognition\\training_sample_list.txt'
    output_dir = ".\\Speaker_Recognition\\samples\\"
    model_dir = ".\\Speaker_Recognition\\gmm_models\\"

    with open(training_file_path, 'w') as training_file:
        for i, word in enumerate(['up', 'down', 'left'], start=1):
            print("开始录制-{}".format(i))
            recordVoice(word, speakerName)
            training_file.write("{}-2024\\{}_{}.wav\n".format(speakerName, speakerName, word))

    features = np.asarray(())
    with open(training_file_path, 'r') as file_paths:
        for count, path in enumerate(file_paths, start=1):
            path = path.strip()
            print("处理文件: {}".format(path))

            sr, audio = read(os.path.join(output_dir, path))
            vector = extract_features(audio, sr, nfft=2048)

            features = np.vstack((features, vector)) if features.size else vector

            if count % 3 == 0:
                gmm = GaussianMixture(n_components=16, max_iter=200, covariance_type='diag', n_init=3)
                gmm.fit(features)

                model_filename = "{}.gmm".format(path.split("-")[0])
                with open(os.path.join(model_dir, model_filename), 'wb') as model_file:
                    cPickle.dump(gmm, model_file)

                print("+ {} 的建模完成，数据点数量 = {}".format(model_filename, features.shape))
                features = np.asarray(())
