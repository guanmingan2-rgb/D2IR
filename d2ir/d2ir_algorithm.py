"""
D2IR 核心算法实现
两阶段优化: 
  Stage 1: 序贯退化估计 (Sequential Degradation Estimation)
  Stage 2: 初始潜在变量优化 (Initial Latent Optimization)
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """从数组提取对应时间步的值，支持numpy和torch"""
    if isinstance(arr, np.ndarray):
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    else:
        res = arr.to(timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class D2IRRestorer:
    """
    D2IR 图像恢复器
    基于预训练扩散模型，在推理阶段通过两阶段优化实现盲图像恢复
    """
    
    def __init__(
        self,
        model,
        diffusion,
        degradation_model,
        device='cuda',
        # Stage 1 参数
        stage1_lr_phi=1e-2,
        stage1_gradient_scale=1.0,
        stage1_use_ddim=False,
        # Stage 2 参数  
        stage2_lr_x=1e-2,
        stage2_K=30,
        stage2_ddim_steps=10,
    ):
        self.model = model
        self.diffusion = diffusion
        self.degradation = degradation_model
        self.device = device
        
        self.stage1_lr_phi = stage1_lr_phi
        self.stage1_gradient_scale = stage1_gradient_scale
        self.stage1_use_ddim = stage1_use_ddim
        
        self.stage2_lr_x = stage2_lr_x
        self.stage2_K = stage2_K
        self.stage2_ddim_steps = stage2_ddim_steps
        
        # 获取扩散调度参数
        self._setup_diffusion_params()
        
    def _setup_diffusion_params(self):
        """设置扩散过程所需的alpha等参数"""
        self.alphas_cumprod = torch.from_numpy(
            self.diffusion.alphas_cumprod
        ).float().to(self.device)
        self.sqrt_alphas_cumprod = torch.from_numpy(
            np.sqrt(self.diffusion.alphas_cumprod)
        ).float().to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.from_numpy(
            np.sqrt(1 - self.diffusion.alphas_cumprod)
        ).float().to(self.device)
        self.sqrt_recip_alphas_cumprod = torch.from_numpy(
            np.sqrt(1.0 / self.diffusion.alphas_cumprod)
        ).float().to(self.device)
        self.sqrt_recipm1_alphas_cumprod = torch.from_numpy(
            np.sqrt(1.0 / self.diffusion.alphas_cumprod - 1)
        ).float().to(self.device)
        self.posterior_variance = torch.from_numpy(
            self.diffusion.posterior_variance
        ).float().to(self.device)
        self.posterior_mean_coef1 = torch.from_numpy(
            self.diffusion.posterior_mean_coef1
        ).float().to(self.device)
        self.posterior_mean_coef2 = torch.from_numpy(
            self.diffusion.posterior_mean_coef2
        ).float().to(self.device)
        self.alpha_bar_prev = torch.from_numpy(
            np.append(1.0, self.diffusion.alphas_cumprod[:-1])
        ).float().to(self.device)
        
        self.num_timesteps = self.diffusion.num_timesteps
        
        # DDIM步长映射
        if hasattr(self.diffusion, 'timestep_map'):
            self.timestep_map = self.diffusion.timestep_map
        else:
            self.timestep_map = list(range(self.num_timesteps))
            
    def _predict_x0_from_eps(self, x_t, t, eps):
        """从x_t和预测的epsilon计算x_0"""
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def _predict_eps_from_x0(self, x_t, t, x0):
        """从x_t和x_0计算epsilon"""
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - x0
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    
    def _q_posterior_mean(self, x_start, x_t, t):
        """计算后验均值 μ_θ(x_t, t)"""
        return (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
    
    def stage1_sequential_degradation_estimation(
        self,
        y,
        num_steps=None,
        progress=True,
    ):
        """
        阶段1: 序贯退化估计
        估计退化算子参数 φ_est
        
        Args:
            y: 退化图像 [B,C,H,W], 范围[-1,1]
            num_steps: 逆向扩散步数，默认使用全部
            progress: 是否显示进度条
            
        Returns:
            phi_est: 估计的退化参数 (即更新后的degradation model state)
        """
        B, C, H, W = y.shape
        shape = (B, C, H, W)
        device = y.device
        
        num_steps = num_steps or self.num_timesteps
        indices = list(range(num_steps))[::-1]  # T 到 1
        
        # 1. 从 N(0,I) 采样 x_T
        x_t = torch.randn(shape, device=device, dtype=y.dtype)
        
        # 优化器用于更新退化参数
        phi_optimizer = torch.optim.Adam(
            self.degradation.parameters(),
            lr=self.stage1_lr_phi
        )
        
        if progress:
            indices = tqdm(indices, desc="Stage 1: Degradation Estimation")
        
        for i in indices:
            t_batch = torch.tensor([i] * B, device=device, dtype=torch.long)
            
            # 需要梯度进行优化
            x_t = x_t.detach().requires_grad_(True)
            
            # 2. 预测 x̂_0|t
            with torch.enable_grad():
                # 使用diffusion的model wrapper以正确映射timestep
                if hasattr(self.diffusion, '_wrap_model'):
                    model_fn = self.diffusion._wrap_model(self.model)
                else:
                    model_fn = self.model
                model_output = model_fn(x_t, t_batch)
                if model_output.shape[1] == 6:  # learn_sigma
                    model_output = model_output[:, :3]
                eps = model_output
                pred_x0 = self._predict_x0_from_eps(x_t, t_batch, eps)
                pred_x0 = pred_x0.clamp(-1, 1)
                
                # 3. 计算一致性损失 L_cons = ||y - D_φ(x̂_0|t)||^2
                degraded_pred = self.degradation(pred_x0)
                L_cons = F.mse_loss(y, degraded_pred)
                
                # 4. 更新退化参数 φ
                phi_optimizer.zero_grad()
                L_cons.backward(retain_graph=True)
                grad_x = x_t.grad.detach()
                phi_optimizer.step()
                x_t.grad.zero_()
            
            # 6. 更新 x_{t-1} ~ N(μ_θ + s·∇L_cons, σ_t^2 I)
            with torch.no_grad():
                posterior_mean = self._q_posterior_mean(pred_x0.detach(), x_t.detach(), t_batch)
                sigma_t = _extract_into_tensor(
                    torch.sqrt(self.posterior_variance + 1e-12), t_batch, x_t.shape
                )
                
                # 引导式更新: mean + s * grad
                guided_mean = posterior_mean + self.stage1_gradient_scale * grad_x.detach()
                
                if i > 0:
                    noise = torch.randn_like(x_t, device=device)
                    x_t = guided_mean + sigma_t * noise
                else:
                    x_t = guided_mean
        
        return self.degradation.state_dict()
    
    def _ddim_reverse_single_step(self, model, x, t, model_kwargs=None):
        """DDIM单步逆向采样: x_t -> x_{t-1}"""
        if model_kwargs is None:
            model_kwargs = {}
            
        out = self.diffusion.p_mean_variance(
            model, x, t, clip_denoised=True,
            model_kwargs=model_kwargs
        )
        pred_x0 = out["pred_xstart"]
        
        # 从pred_x0和x计算eps
        sqrt_recip = _extract_into_tensor(
            self.sqrt_recip_alphas_cumprod, t, x.shape
        )
        sqrt_recipm1 = _extract_into_tensor(
            self.sqrt_recipm1_alphas_cumprod, t, x.shape
        )
        eps = (sqrt_recip * x - pred_x0) / sqrt_recipm1
        
        # DDIM: x_{t-1} = sqrt(alpha_{t-1}) * pred_x0 + sqrt(1-alpha_{t-1}) * eps
        alpha_bar_prev = _extract_into_tensor(self.alpha_bar_prev, t, x.shape)
        
        mean = pred_x0 * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev) * eps
        return mean, pred_x0
    
    def _ddim_reverse_full(self, x_T, model_kwargs=None):
        """
        DDIM完整逆向: x_T -> x_0
        从噪声x_T通过DDIM采样得到清晰图像x_0
        """
        if model_kwargs is None:
            model_kwargs = {}
            
        B = x_T.shape[0]
        device = x_T.device
        x = x_T
        
        # 使用DDIM的步长
        if hasattr(self.diffusion, 'timestep_map'):
            indices = list(range(len(self.diffusion.timestep_map)))[::-1]
        else:
            indices = list(range(self.num_timesteps))[::-1]
        
        for i in indices:
            t = torch.tensor([i] * B, device=device, dtype=torch.long)
            x, pred_x0 = self._ddim_reverse_single_step(
                self.model, x, t, model_kwargs
            )
        
        return x  # 最终x即为x_0
    
    def stage2_initial_latent_optimization(
        self,
        y,
        phi_est_state=None,
        x_T_init=None,
        progress=True,
    ):
        """
        阶段2: 初始潜在变量优化
        固定退化算子，优化x_T以获得清晰图像
        
        Args:
            y: 退化图像 [B,C,H,W]
            phi_est_state: 阶段1估计的退化参数，None则使用当前参数
            x_T_init: 初始x_T，None则从N(0,I)采样
            progress: 是否显示进度条
            
        Returns:
            x0_pred: 预测的清晰图像 [B,C,H,W]
        """
        B, C, H, W = y.shape
        shape = (B, C, H, W)
        device = y.device
        d = H * W * C  # 潜在空间维度
        
        # 释放阶段1占用的显存
        torch.cuda.empty_cache()
        
        # 固定退化算子
        if phi_est_state is not None:
            self.degradation.load_state_dict(phi_est_state)
        for p in self.degradation.parameters():
            p.requires_grad = False
            
        # 1. 采样初始 x_T
        if x_T_init is None:
            x_T = torch.randn(shape, device=device, dtype=y.dtype)
        else:
            x_T = x_T_init.clone()
        x_T = x_T.requires_grad_(True)
        
        # 使用DDIM步数
        ddim_steps = min(self.stage2_ddim_steps, len(self.diffusion.timestep_map) if hasattr(self.diffusion, 'timestep_map') else self.num_timesteps)
        
        # 创建简化版DDIM reverse用于梯度计算
        def ddim_reverse_fn(x_t):
            """可微分的DDIM reverse，用于梯度传播"""
            x = x_t
            if hasattr(self.diffusion, 'timestep_map'):
                indices = list(range(len(self.diffusion.timestep_map)))[::-1]
            else:
                indices = list(range(self.num_timesteps))[::-1]
            indices = indices[:ddim_steps]  # 使用指定步数
            
            for i in indices:
                t = torch.tensor([i] * B, device=device, dtype=torch.long)
                x, _ = self._ddim_reverse_single_step(self.model, x, t, {})
            return x
        
        # 2. 初始化优化
        x_T_optimizer = torch.optim.Adam([x_T], lr=self.stage2_lr_x)
        
        if progress:
            k_range = tqdm(range(self.stage2_K), desc="Stage 2: Latent Optimization")
        else:
            k_range = range(self.stage2_K)
            
        for k in k_range:
            x_T_optimizer.zero_grad()
            
            # x̂_0^(k) = DDIMReverse(x_T^(k), δt)
            x0_pred = ddim_reverse_fn(x_T)
            
            # 损失: ||y - D_φ_est(x̂_0^(k))||^2
            degraded_pred = self.degradation(x0_pred)
            loss = F.mse_loss(y, degraded_pred)
            loss.backward()
            x_T_optimizer.step()
            
            # 归一化: x_T^(k+1) = (x_T / ||x_T||) * sqrt(d)
            with torch.no_grad():
                norm = x_T.norm(dim=(1,2,3), keepdim=True) + 1e-8
                x_T.data = (x_T.data / norm) * np.sqrt(d)
        
        # 最终: 用优化后的x_T做DDIM得到x̂_0
        with torch.no_grad():
            x_T_final = x_T.detach()
            x0_pred = self._ddim_reverse_full(x_T_final)
            x0_pred = x0_pred.clamp(-1, 1)
            
        return x0_pred
    
    def restore(self, y, progress=True):
        """
        完整的D2IR恢复流程: Stage1 + Stage2
        
        Args:
            y: 退化图像 [B,C,H,W], 范围[-1,1]
            progress: 是否显示进度条
            
        Returns:
            x0_pred: 恢复的清晰图像
        """
        # Stage 1: 估计退化参数
        phi_est = self.stage1_sequential_degradation_estimation(y, progress=progress)
        
        # Stage 2: 优化潜在变量，获得清晰图像
        x0_pred = self.stage2_initial_latent_optimization(
            y, phi_est_state=phi_est, progress=progress
        )
        
        return x0_pred
