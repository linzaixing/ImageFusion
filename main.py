import torch
import torch.nn as nn
import cv2
import numpy as np

torch.set_grad_enabled(False)

class MidValueFilter(torch.nn.Module):
    def __init__(self):
        super(MidValueFilter, self).__init__()

        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv.weight.data = torch.ones(1, 1, 3, 3) / 9  # 将每个权重设为1/9，表示取周围9个像素的平均值
        self.conv.bias.data = torch.zeros_like(self.conv.bias.data)

    def forward(self, x):
        return self.conv(x)

class ImageFusion(torch.nn.Module):
    def __init__(self):
        super(ImageFusion, self).__init__()
        self.filter = MidValueFilter()

    def forward(self, fore_img, back_img, alpha_img):
        # np转tensor并归一化
        fore_tensor = torch.from_numpy(fore_img).float() / 255
        back_tensor = torch.from_numpy(back_img).float() / 255
        alpha_tensor = torch.from_numpy(alpha_img).float() / 255

        # alpha单通道转4D Tensor
        alpha_tensor = alpha_tensor.unsqueeze(0).unsqueeze(0)

        alpha_tensor = self.filter(alpha_tensor)   #对mask掩模进行滤波

        # alpha单通道转三通道
        alpha_pic = np.zeros(fore_tensor.shape, np.uint8)
        alpha_pic = torch.from_numpy(alpha_pic).float()
        alpha_pic[..., 0] = alpha_tensor
        alpha_pic[..., 1] = alpha_tensor
        alpha_pic[..., 2] = alpha_tensor

        # alpha融合
        fusion_tensor = alpha_pic * fore_tensor + (1.0 - alpha_pic) * back_tensor

        fusion_tensor = fusion_tensor * 255
        fusion_img = np.clip(fusion_tensor.numpy().astype('uint8'), 0, 255)
        return fusion_img

if __name__ == "__main__":
    fore_img = cv2.imread('jump.jpg', cv2.IMREAD_UNCHANGED)
    back_img = cv2.imread('seaside.jpg', cv2.IMREAD_UNCHANGED)
    alpha_img = cv2.imread('mask.jpg', cv2.IMREAD_GRAYSCALE)

    fusion = ImageFusion()
    result_img = fusion(fore_img, back_img, alpha_img)
    cv2.imwrite('fusion.jpg', result_img)