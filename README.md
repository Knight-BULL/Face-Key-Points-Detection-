# Face-Key-Points-Detection
Face Key Point Detection  On Win10 With Cafe And VS2015

# 1.数据（datasets）
## 1.1数据说明（Data Description）
### 1.数据有17000多张人脸图片，其中标注文件13000多个。
### 2.每张图有7个关键点，每张图片的标注文件第0个是文件名，1，2是矩形框左上角，3,4是矩形框右下角
6-12 对应点的x轴坐标，12-18对应点的y轴坐标。point0是左眼眼角,point1是左眼右眼角，point2是
右眼左眼角,point3是右眼右眼角，point4是鼻子，point5是左嘴角，point6是右嘴角。
![图片加载失败](https://github.com/thehappysheep/Face-Key-Points-Detection-/blob/master/example.jpg)
# 2编程思想
## 2.1数据检查
检查标注文件每一行是否有19个数据，如果有数据缺失，除去缺失项
## 2.2图片预处理
### 1.根据标签中图片名字读取图片。
### 2.根据标注文件把图片中的人脸剪切出来然后编程灰度图片然后压缩到网络需要的size。
## 2.3关键点数据处理
### 1.裁剪后关键点相对原来的参考点发生变化，相当于平移，把裁剪后的图框左上角作为参考点。
### 2.图片压缩后相对与要来尺寸发生变化要成一个scale，等于压缩尺寸/裁剪图框尺寸。
   修改：直接归一化就好，不需要乘上压缩此村（会导致模型不收敛）
## 2.4 生成HDF5文件
1.读取条标注处理一个图片
## 2.5 生成回归检测文件
因为使用的是windows系统，根据caffe自带的classification.cpp修改成regression.cpp生成可执行文件exe。修改思路：分类和回归网络基本一样，损失函数不同。
测试图片要训练文件一致（训练时候并没用使用减去均值的操作，而是直接除以255做归一化，所对classification类中的函数做修改，删去减去均值处理，增加归一化部分。）
ressioner.cpp思路：指指定计算单元----》2.2caffe——net加载网络模型（包括训练阶段）-----------》net-blob读取数据并reshape---》读取均值生成均值图片
----》预处理图片（灰度，归一化，分通道存储）------》blob加载处理后的图片------》运行net_forward()获取output值
# 3.编程遇到的问题
## 3.1数据处理遇到问题
### 1.编程思想很重要，刚开始我编程生成各种中间文件，处理起来很麻烦，后来使用直接生成目标文件。
### 2.在处理时候洗牌很重要（shuffle），可以在直接在标注数据中打乱顺序，直接根据标注文件读取图片。
### 3.需要从训练数据中随机拿10%-20%作为验证文件。
### 4.生成HDF5文件时候标签和数据要对应，可以提前定义好标签和数据的数据格式使用numpy，
   比如data =np.zeros((datasize,pic_depth,pic_width,pic_high),dtype=np.float32)。等数据全部
  读到定义的数组内在写入HDF5文件中；读一次写一次会导致转化速度特别慢（教训）。
### 5.在train_val_net.prototxt文件中不能直接读取hdf5文件，需要把HDF5地址写到txt文件中，不加“”。
### 6.编程时候注意函数的封装模块化和日志打印。
## 3.2模型训练和测试时候遇到问题
### 1.模型不收敛，图片归一化处没做好。
# 备注：
### 1.caffe中要求1个hdf5文件大小不超过2GB，所以如果数据量太大，建议生成多个hdf5文件。
### 需要注意的是Caffe中HDF的DataLayer不支持transform，所以数据存储前就提前进行了减去均值的步骤
### 2.验证测试集时候把train_val_net.prototxt的两个data layer替换成input_shape，然后去掉最后一层EuclideanLoss就可以了，
     input_shape定：
     input: "data"
     input_shape{
       dim: 1
       dim: 1
       dim: 40
       dim: 40
     }
     实验发现这个结构运行报错，应改成一下这种结构
     layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape: { dim: 1 dim: 1 dim: 40 dim: 40 } }
}
### 3.picture_mark是用来在图片标注的
### 4.需要数据集的可以联系我，已经上传到百度云。

