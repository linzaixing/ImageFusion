import torch
import torch.nn as nn
import cv2
import numpy as np

torch.set_grad_enabled(False)

class AverageFilter(torch.nn.Module):
    def __init__(self):
        super(AverageFilter, self).__init__()

        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv.weight.data = torch.ones(1, 1, 3, 3) / 9  # 将每个权重设为1/9，表示取周围9个像素的平均值
        self.conv.bias.data = torch.zeros_like(self.conv.bias.data)

    def forward(self, x):
        return self.conv(x)

class ImageFusion(torch.nn.Module):
    def __init__(self):
        super(ImageFusion, self).__init__()
        self.filter = AverageFilter()

    def forward(self, fore_tensor, back_tensor, alpha_tensor):

        # 对mask掩模进行均值滤波
        alpha_tensor = self.filter(alpha_tensor)

        alpha_pic = alpha_tensor.expand_as(fore_tensor)

        # alpha融合
        fusion_tensor = alpha_pic * fore_tensor + (1.0 - alpha_pic) * back_tensor
        fusion_tensor = fusion_tensor.squeeze().permute(1, 2, 0)

        fusion_tensor = fusion_tensor * 255
        return fusion_tensor

if __name__ == "__main__":
    fore_img = cv2.imread('jump.jpg', cv2.IMREAD_UNCHANGED)
    back_img = cv2.imread('seaside.jpg', cv2.IMREAD_UNCHANGED)
    alpha_img = cv2.imread('mask.jpg', cv2.IMREAD_GRAYSCALE)

    # np转tensor并归一化
    fore_tensor = torch.from_numpy(fore_img).permute(2, 0, 1).unsqueeze(0).float() / 255
    back_tensor = torch.from_numpy(back_img).permute(2, 0, 1).unsqueeze(0).float() / 255
    alpha_tensor = torch.from_numpy(alpha_img).unsqueeze(0).unsqueeze(1).float() / 255


    fusion = ImageFusion()
    fusion.eval()
    with torch.no_grad():
        result_img = fusion(fore_tensor, back_tensor, alpha_tensor)
    fusion_img = np.clip(result_img.numpy().astype('uint8'), 0, 255)
    cv2.imwrite('fusion.jpg', fusion_img)

    dummy_fore = torch.randn(1, 3, 512, 288, requires_grad=True)
    dummy_back = torch.randn(1, 3, 512, 288, requires_grad=True)
    dummy_alpha = torch.randn(1, 1, 512, 288, requires_grad=True)
    torch.onnx.export(fusion, (dummy_fore, dummy_back, dummy_alpha), "model.onnx")
