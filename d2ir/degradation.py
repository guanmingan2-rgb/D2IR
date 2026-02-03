"""
D2IR: 可学习退化模型模块
Learnable Degradation Model D_φ for blind image restoration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as st


def gkern(kernlen=21, nsig=3):
    """生成2D高斯核"""
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel.astype(np.float32)


class LearnableGaussianBlur(nn.Module):
    """
    可学习高斯模糊退化模型 D_φ
    参数φ为高斯核，通过优化进行估计
    输入输出范围: [0, 1] (与扩散模型[-1,1]转换在外部处理)
    """
    def __init__(self, kernel_size=21, init_nsig=3.0):
        super().__init__()
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1, 'kernel size must be odd'
        
        # 使用可学习参数初始化高斯核
        init_kernel = gkern(kernel_size, init_nsig)
        # 将核展平为可学习参数 (kernel_size^2)
        self.kernel_params = nn.Parameter(
            torch.FloatTensor(init_kernel).view(-1),
            requires_grad=True
        )
        
    def _get_kernel(self):
        """获取归一化的2D卷积核 (非负且和为1)"""
        k = self.kernel_params.view(self.kernel_size, self.kernel_size)
        k = F.softplus(k) + 1e-6  # 确保非负
        k = k / k.sum()  # 归一化
        return k.unsqueeze(0).unsqueeze(0)  # 1x1xKxK
        
    def forward(self, x):
        """
        x: [B, C, H, W], 范围 [0, 1]
        对每个通道分别卷积
        """
        padding = self.kernel_size // 2
        x = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        
        weight = self._get_kernel()
        weight = weight.expand(x.shape[1], 1, self.kernel_size, self.kernel_size)
        
        out = F.conv2d(x, weight, groups=x.shape[1])
        return out


class DegradationWrapper(nn.Module):
    """
    退化模型包装器，处理扩散模型的[-1,1]范围与退化模型的[0,1]范围转换
    D_φ: 清晰图像 x ∈ [-1,1] -> 退化图像 y ∈ [-1,1]
    """
    def __init__(self, degradation_model):
        super().__init__()
        self.degradation = degradation_model
        
    def forward(self, x):
        """
        x: 清晰图像 [B,C,H,W], 范围 [-1, 1]
        返回: 退化图像, 范围 [-1, 1]
        """
        # [-1, 1] -> [0, 1]
        x_01 = (x + 1) / 2
        # 退化
        y_01 = self.degradation(x_01)
        # [0, 1] -> [-1, 1]
        return y_01 * 2 - 1


def create_learnable_blur(kernel_size=21, init_nsig=3.0, device='cuda'):
    """创建可学习高斯模糊退化模型"""
    blur = LearnableGaussianBlur(kernel_size=kernel_size, init_nsig=init_nsig)
    return DegradationWrapper(blur).to(device)
