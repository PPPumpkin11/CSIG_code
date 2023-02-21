customize_service.py
一、SelectNetworkInput函数
作用是从输入的序列，也就是npy文件中随机抽取固定的帧数（本次任务要求是60帧）来进行后面的处理
npy文件中其实是姿态的三元轴角式，在这个函数中，我们除了进行固定帧数的帧的抽取，我们还将三元轴角式转换为旋转矩阵，然后转换为rot6D格式，最终作为网络的输入
二、PTVisionService类
1._preprocess函数
①加载npy数据
②用上面的SelectNetworkInput函数进行抽帧和数据转换处理
③将数据变为pytorch中的一个结点，可以进行梯度反向传播
④将数据加载到gpu

2._inference函数
包含我们最终用于推理的model，经过model产生输出

三、ActionRecogModel函数
定义model = Net(ActionLength)。为了上面_inference函数可以调用



Net.py
是baseline模型，只是用了一个简单的神经网络
可以看到，这个神经网络的输入是ActionLength * 24 * 6，
也就是所有被我们抽取到的视频帧的npy文件（那个72维的参数）；
输出是10也就是我们一开始基础赛的10类动作
后面根据这个baseline修改为自己的模型



utils.py
1.matrix_to_rotation_6d函数
将旋转矩阵转为6D旋转表示
旋转矩阵是3*3，也就是原来是（x，3，3），现在要变为（y，6）
为什么有旋转矩阵？
在spml模型中，将人体分为24个节点，以0为根节点，通过其它23个节点相对其父节点的旋转角度可以定义出一个人的姿态。
节点相对其父节点的旋转本来是用轴角式表示的，一般来说轴角式是一个四元组(x,y,z,θ)，
但是spml文章中用三元组来表示，即θ=(x,y,z)
但是轴角式并不方便计算，所以一般会将其转化为旋转矩阵来计算，所以参数数量会从3个变为9个
主要过程：轴角式四元组(x,y,z,θ)→轴角式三元组θ=(x,y,z)→旋转矩阵（...，3，3）→6D旋转表示（...，6）

2.quaternion_to_matrix函数
将四元组转为旋转矩阵表示
原来是（x，4），现在要变为（x，3，3）

3.axis_angle_to_matrix函数
将轴角式三元组转为旋转矩阵表示
原来是（x，3），现在要变为（x，3，3）

4.axis_angle_to_quaternion函数
将轴角式三元组转为轴角式四元组
原来是（x，3），现在要变为（x，4）

调用的时候，也仅是quaternion_to_matrix函数和axis_angle_to_quaternion函数
使用到了（可能是抽出来的时候已经是三元组了所以不需要四元组→三元组）