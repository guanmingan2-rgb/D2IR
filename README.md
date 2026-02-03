# D2IR: Diffusion-based Degradation Estimation and Image Restoration

基于扩散模型的盲图像恢复方法，在推理阶段通过两阶段优化实现，**无需训练**，仅使用预训练扩散模型。

## 方法概述

D2IR 采用两阶段优化策略：

### 阶段 1: 序贯退化估计 (Sequential Degradation Estimation)
- **目标**: 估计准确的退化算子参数 φ_est
- **输入**: 退化图像 y、可学习退化模型 D_φ、预训练扩散模型 ε_θ
- **过程**: 在逆向扩散过程中，每步同时更新退化参数 φ 和潜在变量 x_t
  - 预测 x̂_0|t = (x_t - √(1-ā_t) ε_θ(x_t,t)) / √ā_t
  - 一致性损失 L_cons = ||y - D_φ(x̂_0|t)||²
  - 更新 φ: φ ← φ - η_φ ∇_φ L_cons
  - 更新 x_{t-1}: 引导式采样

### 阶段 2: 初始潜在变量优化 (Initial Latent Optimization)
- **目标**: 获得预测的清晰图像 x̂_0
- **输入**: 固定阶段1估计的退化参数 φ_est
- **过程**: 优化初始潜在变量 x_T
  - x̂_0(x_T) = DDIMReverse(x_T, δt)
  - 最小化 ||y - D_φ_est(x̂_0)||²
  - 对 x_T 进行归一化约束

## 项目结构

```
D2IR/
├── d2ir/
│   ├── __init__.py
│   ├── degradation.py      # 可学习退化模型
│   └── d2ir_algorithm.py   # 两阶段算法核心实现
├── run_d2ir.py            # 主入口脚本
├── requirements.txt
└── README.md
```

## 依赖

- PyTorch >= 1.9.0
- 依赖 GenerativeDiffusionPrior (GDP) 项目的 guided_diffusion 模块

## 安装

```bash
# 1. 安装 GDP 的 guided-diffusion 包
cd ../GenerativeDiffusionPrior
pip install -e .

# 2. 安装 D2IR 依赖
cd ../D2IR
pip install -r requirements.txt
```

## 预训练模型

需要下载 ImageNet-256 预训练 DDPM 模型，参考 [Guided Diffusion](https://github.com/openai/guided-diffusion) 或 GDP 的 README。

将模型放置在 `GenerativeDiffusionPrior/scripts/models/256x256_diffusion_uncond.pt`

## 使用方法

### 单张图像恢复

```bash
# 从退化图像恢复
python run_d2ir.py --input_path /path/to/degraded.png --output_dir ./results

# 测试模式：从清晰图像创建退化版本后恢复
python run_d2ir.py --input_path /path/to/clean.png --create_degraded --output_dir ./results
```

### 完整参数示例

```bash
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 \
  --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 \
  --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True \
  --use_scale_shift_norm True"

python run_d2ir.py $MODEL_FLAGS --timestep_respacing ddim50 \
  --input_path ./test_image.png --output_dir ./results \
  --stage1_steps 50 --stage2_K 30 --stage2_ddim_steps 10 \
  --model_path /path/to/256x256_diffusion_uncond.pt
```

### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --stage1_steps | 50 | 阶段1扩散步数 |
| --stage1_lr | 1e-2 | 阶段1退化参数学习率 |
| --stage1_scale | 1.0 | 阶段1梯度引导尺度 |
| --stage2_K | 30 | 阶段2潜在优化迭代数 |
| --stage2_lr | 1e-2 | 阶段2潜在变量学习率 |
| --stage2_ddim_steps | 10 | 阶段2 DDIM 逆向步数 |
| --kernel_size | 21 | 退化核大小 |
| --use_fp16 | True | 将模型转换为FP16以降低显存占用 |

## 与 GDP 的关系

D2IR 以 [Generative Diffusion Prior (GDP)](../GenerativeDiffusionPrior) 为 baseline，主要扩展：

1. **可学习退化模型**: 将固定高斯模糊核改为可学习参数，在推理时估计
2. **两阶段优化**: 先估计退化，再优化反演
3. **纯推理优化**: 不进行任何训练，仅使用预训练模型

## 参考文献

- GDP: Generative Diffusion Prior for Unified Image Restoration and Enhancement
- Guided Diffusion
- DDRM
