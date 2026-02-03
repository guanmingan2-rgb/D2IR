#!/usr/bin/env python3
"""
D2IR 主入口脚本
Diffusion-based Degradation Estimation and Image Restoration
基于GDP的盲图像恢复，推理时两阶段优化，无需训练
"""

import argparse
import os
import sys

# 添加GDP路径以导入guided_diffusion
GDP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'GenerativeDiffusionPrior', 'scripts')
sys.path.insert(0, GDP_PATH)

import numpy as np
import torch as th
from PIL import Image
from tqdm import tqdm

# 导入GDP组件
from guided_diffusion import logger
from guided_diffusion.script_util_x0 import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

# 导入D2IR组件
from d2ir import create_learnable_blur, D2IRRestorer


def load_image(path, image_size=256, device='cuda'):
    """加载并预处理图像"""
    img = Image.open(path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0
    img = (img * 2 - 1)  # 转到 [-1, 1]
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = th.from_numpy(img).float().unsqueeze(0).to(device)
    
    # Resize到目标尺寸
    if img.shape[2] != image_size or img.shape[3] != image_size:
        img = th.nn.functional.interpolate(
            img, size=(image_size, image_size),
            mode='bilinear', align_corners=False
        )
    return img


def save_image(tensor, path):
    """保存图像张量到文件"""
    img = ((tensor + 1) * 127.5).clamp(0, 255).byte()
    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(img).save(path)


def create_degraded_image(clean_img, kernel_size=9, sigma=2.0):
    """创建退化图像 (用于测试，模拟真实退化)"""
    from scipy.ndimage import gaussian_filter
    img_np = ((clean_img.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2)
    degraded = np.zeros_like(img_np)
    for c in range(3):
        degraded[:, :, c] = gaussian_filter(img_np[:, :, c], sigma=sigma)
    degraded = th.from_numpy(degraded).float().permute(2, 0, 1).unsqueeze(0)
    degraded = degraded * 2 - 1
    return degraded.to(clean_img.device)


def main():
    parser = argparse.ArgumentParser(description='D2IR: Blind Image Restoration')
    defaults = dict(
        clip_denoised=True,
        batch_size=1,
        use_ddim=True,
        timestep_respacing="ddim50",
        model_path=os.path.join(
            os.path.dirname(__file__), '..', 'BIRD', 'checkpoints',
            '256x256_diffusion_uncond.pt'
        ),
    )
    defaults.update(model_and_diffusion_defaults())
    # 与GDP示例一致的模型配置，确保与ImageNet-256预训练权重匹配
    gdp_model_defaults = dict(
        attention_resolutions="32,16,8",
        class_cond=False,
        diffusion_steps=1000,
        image_size=256,
        learn_sigma=True,
        noise_schedule="linear",
        num_channels=256,
        num_head_channels=64,
        num_res_blocks=2,
        resblock_updown=True,
        use_fp16=True,
        use_scale_shift_norm=True,
    )
    defaults.update(gdp_model_defaults)
    add_dict_to_argparser(parser, defaults)
    
    # D2IR 特定参数
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--input_path", required=True, help='输入退化图像路径')
    parser.add_argument("--output_dir", default='./results', help='输出目录')
    parser.add_argument("--create_degraded", action='store_true', 
                        help='从输入图像创建退化版本(用于测试)')
    parser.add_argument("--stage1_steps", default=50, type=int, help='阶段1扩散步数')
    parser.add_argument("--stage1_lr", default=1e-2, type=float, help='阶段1退化参数学习率')
    parser.add_argument("--stage1_scale", default=1.0, type=float, help='阶段1梯度尺度')
    parser.add_argument("--stage2_K", default=30, type=int, help='阶段2潜在优化迭代数')
    parser.add_argument("--stage2_lr", default=1e-2, type=float, help='阶段2潜在变量学习率')
    parser.add_argument("--stage2_ddim_steps", default=10, type=int, help='阶段2 DDIM 逆向步数')
    parser.add_argument("--kernel_size", default=21, type=int, help='退化核大小')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = th.device('cuda')
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger.configure(dir=args.output_dir)
    
    # 1. 加载预训练扩散模型
    logger.log("Loading diffusion model...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(th.load(args.model_path, map_location="cpu"))
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    
    # 2. 创建可学习退化模型
    degradation = create_learnable_blur(
        kernel_size=args.kernel_size,
        init_nsig=3.0,
        device=device
    )
    
    # 3. 创建D2IR恢复器
    restorer = D2IRRestorer(
        model=model,
        diffusion=diffusion,
        degradation_model=degradation,
        device=device,
        stage1_lr_phi=args.stage1_lr,
        stage1_gradient_scale=args.stage1_scale,
        stage2_lr_x=args.stage2_lr,
        stage2_K=args.stage2_K,
        stage2_ddim_steps=args.stage2_ddim_steps,
    )
    
    # 4. 加载输入图像
    logger.log("Loading input image...")
    if args.create_degraded:
        # 从输入创建退化版本 (测试用)
        clean_img = load_image(args.input_path, args.image_size, device)
        y = create_degraded_image(clean_img, kernel_size=9, sigma=2.0)
        save_image(clean_img, os.path.join(args.output_dir, 'gt.png'))
    else:
        y = load_image(args.input_path, args.image_size, device)
    
    save_image(y, os.path.join(args.output_dir, 'degraded.png'))
    
    # 5. 运行D2IR恢复
    logger.log("Running D2IR restoration...")
    # Stage 1 需要梯度进行退化参数优化
    phi_est = restorer.stage1_sequential_degradation_estimation(
        y, num_steps=args.stage1_steps, progress=True
    )
    # Stage 2 优化潜在变量
    x0_pred = restorer.stage2_initial_latent_optimization(
        y, phi_est_state=phi_est, progress=True
    )
    
    # 6. 保存结果
    output_path = os.path.join(args.output_dir, 'restored.png')
    save_image(x0_pred, output_path)
    logger.log(f"Saved restored image to {output_path}")


if __name__ == "__main__":
    main()
