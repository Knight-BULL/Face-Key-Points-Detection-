"""
data:2018.8.16
__author__ ='__main__'
"""
import numpy as np
import os
import pandas as pd
from sklearn.utils import shuffle
import h5py

def csvToHdf5():
    TRAIN_CSV = "E:/dataset/FacialPoints/kaggleFaceKeyPoints/training/training.csv"
    datafram = pd.read_csv(os.path.expanduser(TRAIN_CSV))
    datafram['Image'] = datafram['Image'].apply(lambda img: np.fromstring(img, sep=' '))#sep
    #lamdb定义一个匿名函数img ：后面是函数体，np.fromstring(),使用字符串创造矩阵
    datafram = datafram.dropna()
    data = np.vstack(datafram['Image'].values)/255
    label = datafram[datafram.columns[:-1]].values
    print("label.iloc[:,0].size", len(label))
    label = (label-48)/48
    data, label = shuffle(data, label, random_state=0)
    return data, label

if __name__ == "__main__":
    TRAIN_H5_PATH = "E:/dataset/FacialPoints/kaggleFaceKeyPoints/train.hd5"
    VAL_H5_PATH = "E:/dataset/FacialPoints/kaggleFaceKeyPoints/val.hd5"
    data, label = csvToHdf5()
    data = data.reshape(-1, 1, 96, 96)
    dataTrain = data[:-100, :, :, :]
    dataVal = data[-100:, :, :, :]

    label = label.reshape(-1, 1, 1, 30)
    labelTrain = label[:-100, :, :, :]
    labelVal = label[-100:, :, :, :]

    with h5py.File(TRAIN_H5_PATH, 'w') as THdf5File:
        THdf5File.create_dataset('data', data=dataTrain, compression='gzip',compression_opts=4)
        THdf5File.create_dataset('label', data=labelTrain)
        print("训练数据转换结束")

    with h5py.File(VAL_H5_PATH, 'w') as VHdf5File:
        VHdf5File.create_dataset('data', data=dataVal,compression='gzip',compression_opts=4)
        VHdf5File.create_dataset('label', data=labelVal)
        print("验证数据转换结束")

