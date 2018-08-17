"""
Data Description:
每张图右7个关键点
标注文件（19）第0个是文件名，1，2是矩形框左上角，3,4是矩形框右下角6-12 对应点的坐标
12-18对应点的坐标。point0是左眼眼角,point1是左眼右眼角，point2是右眼左眼角,point3是右眼右眼角，
point4是鼻子，point5是左嘴角，point6是右嘴角。

一.数据准备
1抽取没有标注数据的作为测试集
2.在有标注信息的图片里面抽取验证集
二.数据预处理（先测试集后验证集）
1.抠图后标注坐标平移变换
2.resize后坐标再次更新，尺度上变小需要在W和H上乘以一个Scal系数
3.label和图片的数据格式要float32 而且图片和label转成np.arry格式（符合（1,1,40,40）的格式）
三.模型训练
1.准备网络模型train_val_net.prototxt
2.准备超参数solver.prototxt文件
3.准备训练脚本train.bat
4.准备回归预测脚本Regression.bat
5准备py脚本子在测试图片上把关键带点标注出来
"""
import cv2
import os
import numpy as np

'''
data：2018.8.7
__author__ == "syy"
function：读取图片画出对应框和关键点
'''
def show_box_point(test_pic_path,test_list_path,pic_save_path):
    with open(test_list_path, 'r') as f:
        lines = f.readlines()
    keypoint = np.zeros(14, dtype=int)
    for line in lines:
        new_line = line.split()  # str.split(str="", num=string.count(str)).str -- 分隔符，默认为所有的空字符，
        pic_name = new_line[0]
        print(pic_name)
        print(os.path.join(test_pic_path, pic_name))
        x1 = int(new_line[1])
        y1 = int(new_line[2])
        x2 = int(new_line[3])
        y2 = int(new_line[4])
        keypoint[0] = float(new_line[5])
        keypoint[1] = float(new_line[6])
        keypoint[2] = float(new_line[7])
        keypoint[3] = float(new_line[8])
        keypoint[4] = float(new_line[9])
        keypoint[5] = float(new_line[10])
        keypoint[6] = float(new_line[11])
        keypoint[7] = float(new_line[12])
        keypoint[8] = float(new_line[13])
        keypoint[9] = float(new_line[14])
        keypoint[10] = float(new_line[15])
        keypoint[11] = float(new_line[16])
        keypoint[12] = float(new_line[17])
        keypoint[13] = float(new_line[18])
        # 关键点的数组
        img = cv2.imread(os.path.join(test_pic_path, pic_name))
        cv2.imshow("picture", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        img_new = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        for PoiNum in range(7):
            img_new = cv2.circle(img_new, (int(keypoint[PoiNum]), int(keypoint[PoiNum + 7])), 5, (0, 0, 255), -1)
            cv2.putText(img_new, str(PoiNum), (int(keypoint[PoiNum]), int(keypoint[PoiNum + 7])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)#照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
        cv2.imshow('pic_name', img_new)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(os.path.join(pic_save_path, 'mark' + pic_name),  img_new)
    print("picture saved ok!!!!!!!!!!", pic_name)

if __name__ == "__main__":
    pic_dir = "E:/dataset/FacialPoints/pyproject/test/"
    pic_list = "E:/dataset/FacialPoints/pyproject/test.txt"
    pic_save = "E:/dataset/FacialPoints/pyproject/save"
    show_box_point(pic_dir, pic_list, pic_save)

       


