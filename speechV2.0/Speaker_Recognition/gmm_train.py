import os
import pickle
import warnings
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture
from mfcc_coeff import extract_features

warnings.filterwarnings("ignore")


def train_gmm_models(source_dir, dest_dir, train_list_file):
    """
    为每个说话者的语音样本训练GMM模型，并将模型保存到指定目录。

    参数:
    source_dir (str): 语音样本的源目录
    dest_dir (str): 保存GMM模型的目标目录
    train_list_file (str): 包含训练样本文件路径的文件
    """
    # 确保目标目录存在
    os.makedirs(dest_dir, exist_ok=True)

    # 读取训练样本列表文件
    with open(train_list_file, 'r') as file_paths:
        paths = file_paths.readlines()

    # 初始化变量
    features = []
    speaker_count = 0

    # 处理训练样本列表中的每个文件
    for path in paths:
        path = path.strip()
        file_path = os.path.join(source_dir, path)

        try:
            # 读取WAV文件
            sr, audio = read(file_path)

            # 提取MFCC特征
            mfcc_features = extract_features(audio, sr)
            features.append(mfcc_features)

            speaker_count += 1

            # 每处理3个样本后训练一次GMM模型
            if speaker_count == 3:
                all_features = np.vstack(features)

                # 使用16个组件训练GMM模型
                gmm = GaussianMixture(n_components=16, covariance_type='diag', n_init=3)
                gmm.fit(all_features)

                # 保存GMM模型
                speaker_name = path.split("-")[0]
                model_path = os.path.join(dest_dir, "{}.gmm".format(speaker_name))
                with open(model_path, 'wb') as model_file:
                    pickle.dump(gmm, model_file)

                print("+ 为说话者 {} 完成建模，数据点数量 = {}".format(speaker_name, all_features.shape[0]))

                # 重置变量以处理下一个说话者
                features = []
                speaker_count = 0

        except Exception as e:
            print("处理文件 {} 时出错：{}".format(file_path, e))

    # 检查是否有剩余样本需要处理（如果最后一个说话者少于3个样本）
    if speaker_count > 0:
        all_features = np.vstack(features)
        gmm = GaussianMixture(n_components=16, covariance_type='diag', n_init=3)
        gmm.fit(all_features)

        speaker_name = path.split("-")[0]
        model_path = os.path.join(dest_dir, "{}.gmm".format(speaker_name))
        with open(model_path, 'wb') as model_file:
            pickle.dump(gmm, model_file)

        print("+ 为说话者 {} 完成建模，数据点数量 = {}".format(speaker_name, all_features.shape[0]))


if __name__ == "__main__":
    # 定义源目录、目标目录和训练样本列表文件
    source_dir = "samples/"
    dest_dir = "gmm_models/"
    train_list_file = "training_sample_list.txt"

    # 调用函数进行训练
    train_gmm_models(source_dir, dest_dir, train_list_file)
