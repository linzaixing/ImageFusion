# ImageFusion
pytorch设计网络结构实现两张图片alpha的加权融合
具体要求如下：输入：前景图片(U8C3)，背景图片(U8C3),alpha图片(U8C1)；
操作：现在网络中对alpha图片进行中值滤波，滤波器权重为1*1*3*3中值滤波的convolution，然后前景和背景根据mask值加权；
输出：alpha加权融合后的图片
