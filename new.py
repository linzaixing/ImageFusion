import torch
import torch.nn as nn
import cv2
import numpy as np


class ImageFusion(nn.Module):
    def __init__(self):
        super(ImageFusion, self).__init__()
        # 实现均值滤波器
        self.avgpool = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1, bias=False)
        # 设置滤波器权重为均值滤波
        self.avgpool.weight.data.fill_(1.0 / 9.0)

    def forward(self, fore, back, alpha):
        # 对alpha图像进行均值滤波
        alpha = self.avgpool(alpha)
        # 根据alpha对前景和背景图像进行融合
        return alpha * fore + (1 - alpha) * back


# 确保从这里直接运行脚本
if __name__ == "__main__":
    # 读取图像
    fore_img = cv2.imread('jump.jpg', cv2.IMREAD_UNCHANGED)
    back_img = cv2.imread('seaside.jpg', cv2.IMREAD_UNCHANGED)
    alpha_img = cv2.imread('mask.jpg', cv2.IMREAD_GRAYSCALE)

    # 将图像数据转化为tensor
    fore_tensor = torch.from_numpy(fore_img).permute(2, 0, 1).unsqueeze(0).float() / 255
    back_tensor = torch.from_numpy(back_img).permute(2, 0, 1).unsqueeze(0).float() / 255
    alpha_tensor = torch.from_numpy(alpha_img).unsqueeze(0).unsqueeze(1).float() / 255

    # 创建图像融合模型，进行alpha均值滤波和图像融合
    fusion = ImageFusion()
    with torch.no_grad():
        result_img = fusion(fore_tensor, back_tensor, alpha_tensor)

    # result_img已经是融合后的图像，但是保存图像需要将其转换回numpy数组，并且scale到0~255范围
    result_img = (result_img.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # 保存融合后的图像
    cv2.imwrite('fusion.jpg', result_img)

    # 设置假的输入数据，并将模型导出为onnx格式
    dummy_fore = torch.randn(1, 3, 512, 288)
    dummy_back = torch.randn(1, 3, 512, 288)
    dummy_alpha = torch.randn(1, 1, 512, 288)
    torch.onnx.export(fusion, (dummy_fore, dummy_back, dummy_alpha), "model.onnx")