import ipdb  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from ray_diffusion.utils.visualization import plot_to_image


class NoiseScheduler(nn.Module):
    def __init__(
        self,
        max_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        cos_power=2,
        num_inference_steps=100,
        type="linear",
    ):
        super().__init__()
        self.max_timesteps = max_timesteps
        self.num_inference_steps = num_inference_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.cos_power = cos_power
        self.type = type

        if type == "linear":
            self.register_linear_schedule()
        elif type == "cosine":
            self.register_cosine_schedule(cos_power)

        self.inference_timesteps = self.compute_inference_timesteps()

    def register_linear_schedule(self):
        self.register_buffer(
            "betas",
            torch.linspace(
                self.beta_start,
                self.beta_end,
                self.max_timesteps,
                dtype=torch.float32,
            ),
        )
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))

    def register_cosine_schedule(self, cos_power, s=0.008):
        timesteps = (
            torch.arange(self.max_timesteps + 1, dtype=torch.float32)
            / self.max_timesteps
        )
        alpha_bars = (timesteps + s) / (1 + s) * np.pi / 2
        alpha_bars = torch.cos(alpha_bars).pow(cos_power)
        alpha_bars = alpha_bars / alpha_bars[0]
        betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

        self.register_buffer(
            "betas",
            betas,
        )
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))

    def compute_inference_timesteps(
        self, num_inference_steps=None, num_train_steps=None
    ):
        # based on diffusers's scheduling code
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        if num_train_steps is None:
            num_train_steps = self.max_timesteps
        step_ratio = num_train_steps // num_inference_steps
        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].astype(int)
        )
        return timesteps

    def plot_schedule(self, return_image=False):
        fig = plt.figure(figsize=(6, 4), dpi=100)
        alpha_bars = self.alphas_cumprod.cpu().numpy()
        plt.plot(np.sqrt(alpha_bars))
        plt.grid()
        if self.type == "linear":
            plt.title(
                f"Linear (T={self.max_timesteps}, S={self.beta_start}, E={self.beta_end})"
            )
        else:
            self.type == "cosine"
            plt.title(f"Cosine (T={self.max_timesteps}, P={self.cos_power})")
        if return_image:
            image = plot_to_image(fig)
            plt.close(fig)
            return image
