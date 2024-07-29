import numpy as np
from sklearn.preprocessing import scale
import python_speech_features as mfcc


def calculate_delta(features, N=2):
    """
    计算输入特征矩阵的Delta特征。

    参数:
    - features: 输入的特征矩阵，形状为(rows, num_features)。
    - N: 用于计算Delta的窗口大小，默认值为2。

    返回:
    - deltas: Delta特征矩阵，形状与输入特征矩阵相同。
    """
    rows, cols = features.shape
    deltas = np.zeros_like(features)

    for i in range(rows):
        delta_sum = np.zeros(cols)
        for n in range(1, N + 1):
            if i - n < 0:
                first = 0
            else:
                first = i - n
            if i + n >= rows:
                second = rows - 1
            else:
                second = i + n

            delta_sum += n * (features[second] - features[first])

        deltas[i] = delta_sum / (2 * sum([n ** 2 for n in range(1, N + 1)]))

    return deltas


def extract_features(audio, rate, nfft=2048):
    """
    从音频信号中提取MFCC特征和Delta特征，并标准化特征。

    参数:
    - audio: 输入的音频信号。
    - rate: 音频的采样率。
    - nfft: 用于FFT的大小，默认值为2048。

    返回:
    - combined: 组合的特征向量，包含MFCC特征和Delta特征。
    """
    mfcc_features = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, nfft=nfft, appendEnergy=True)
    # 对特征进行标准化
    mfcc_features = scale(mfcc_features)
    # 计算Delta特征
    delta_features = calculate_delta(mfcc_features)
    # 合并MFCC特征和Delta特征
    combined_features = np.hstack((mfcc_features, delta_features))

    return combined_features
