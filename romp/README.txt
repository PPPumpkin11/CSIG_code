base_predictor.py


video.py
对每一个视频产生一个npy文件
一、video_processor类
1. toframe函数
    首先确保输入的路径是字符串
    然后从路径中取出文件名和扩展名，使用了os.path.splitext函数
    如果扩展名不是我们可以处理的类别那就给出提示让用户自己去处理
    然后就使用已经写好的video2frame将视频转为帧
    最后返回文件名和视频帧
    是为了方便后面的调用
2.process_video
  首先对输入的路径进行判断，如果是一个文件夹，则说明已经经过处理，文件夹中应该就是每一帧，所以直接调用collect_image_list函数得到帧的列表；
  如果不是一个文件夹，那么是一个视频，就要调用toframe函数来处理为一帧一帧的；
  不存在则报错
  建立起计时器、数据加载器。
  按照batch size遍历数据集，其中outputs是输入的数据进行数据并行化的结果，results就是经过spml模型提取的参数的结果。
  进行一系列处理，把poses参数提出来并且reshape成（1，72）的，再把同一个视频里面的所有帧的poses参数堆叠成为npy文件，最后保存npy文件
3.相对于官方给出的代码做的修改
  由于所有if语句后面的变量都是false，所以if语句都不运行
  我们的目的是要将一个视频中所有帧对应的那个（1，72）的参数也就是poses堆叠起来成为npy文件（这个是参考示例数据的npy文件得出的结论）
  观察到前面经过spml模型产生的results是一个字典，字典的key是每一帧，value是12个姿态参数，所以需要用代码将poses参数提取出来


ChangeGlobalOrientation2forward.py
用于将不是正面角度的npy文件进行根关节朝向调整。


修改说明.txt


