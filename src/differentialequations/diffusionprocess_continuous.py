import torch

from src.differentialequations.diffusionprocess import DiffusionProcess


class DiffusionProcess_Continuous(DiffusionProcess):
    def __init__(
        self,
        sde_type="VP",
        sde_info={
            "VP": {"beta_min": 0.1, "beta_max": 20},
            "subVP": {"beta_min": 0.1, "beta_max": 20},
            "VE": {"sigma_min": 0.01, "sigma_max": 50},
        },
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert self.discrete is False, "DDPM is only for continuous data"
        self.dt = 1.0 / self.total_steps  # step size
        self.sde = SDE(self.total_steps, sde_type, sde_info)

    def forward_one_step(self, x_prev, t):
        """
        Discretized forward SDE process for actual compuatation:
        x_{t+1} = x_t + f_t(x_t) * dt + G_t * z_t * sqrt(dt)
        """
        f_t, g_t = self.sde.drifts(x_prev, t - 1)
        z = torch.randn_like(x_prev)
        x_t = x_prev + f_t * self.dt + g_t * z * np.sqrt(self.dt)
        return x_t

    def backward_one_step(self, x_t, t, pred_score, clip_denoised=True):
        """
        Discretized backward SDE process for actual compuatation:
        x_{t-1} = x_t - (f_t(x_t) - (G_t)^2 * pred_score) * dt + G_t * z_t * sqrt(dt)
        """
        z = torch.randn_like(x_t).to(device)
        f_t, g_t = self.sde.drifts(x_t, t)
        f_t = f_t.to(device)
        x_prev = (
            x_t - (f_t - g_t**2 * pred_score) * self.dt + g_t * z * np.sqrt(self.dt)
        )
        if clip_denoised and x_t.ndim > 2:
            x_prev.clamp_(-1.0, 1.0)

        return x_prev

    @torch.no_grad()
    def sample(self, noise, net):
        """
        Sample from backward diffusion process
        """
        x_t = noise
        trajs = [x_t]

        for t in reversed(range(1, self.total_steps + 1)):
            pred_score = net(x_t, t)
            x_t = self.backward_one_step(x_t, t, pred_score)
            trajs.append(x_t)
        return x_t, torch.hstack(trajs)

    def forward_sample(self, data):
        trajs = torch.zeros([len(data), self.total_steps + 1])
        x = data.to(device)
        trajs[:, 0] = x
        for t in range(1, self.total_steps + 1):
            x = self.forward_one_step(x, t)
            trajs[:, t] = x
        return x, trajs

    def backward_sample(self, noise, net):
        """
        Sample from backward diffusion process
        """
        x_t = noise
        trajs = [x_t]

        for t in reversed(range(1, self.total_steps + 1)):
            pred_score = net(x_t, t)
            x_t = self.backward_one_step(x_t, t, pred_score)
            trajs.append(x_t)
        return x_t, torch.hstack(trajs)


import numpy as np
import torch


class SDE:
    def __init__(self, total_steps, sde_type="VP", sde_info=None):
        self.total_steps = total_steps
        self.sde_type = sde_type
        assert sde_type in ["VP", "subVP", "VE"]
        self.sde_info = sde_info[sde_type]
        self.coef = self.compute_coef()

    def compute_coef(self):
        if self.sde_type == "VP" or self.sde_type == "subVP":
            beta_0 = self.sde_info["beta_min"]
            beta_1 = self.sde_info["beta_max"]
            return {"beta_0": beta_0, "beta_1": beta_1}

        if self.sde_type == "VE":
            sigma_min = self.sde_info["sigma_min"]
            sigma_max = self.sde_info["sigma_max"]
            return {"sigma_min": sigma_min, "sigma_max": sigma_max}

    def drifts(self, x_t, t: int):
        """
        Compute drifts f_t and g_t for SDE process for each timestep t
        """
        t = t / self.total_steps  # t in [0, 1]
        if self.sde_type == "VP" or self.sde_type == "subVP":
            beta_t = self.coef["beta_0"] + t * (
                self.coef["beta_1"] - self.coef["beta_0"]
            )
            f_t = -0.5 * beta_t * x_t

            if self.sde_type == "VP":
                g_t = np.sqrt(beta_t)
                return f_t, g_t
            if self.sde_type == "subVP":
                discount = 1.0 - np.exp(
                    -2 * self.coef["beta_0"] * t
                    - (self.coef["beta_1"] - self.coef["beta_0"]) * t**2
                )
                g_t = np.sqrt(beta_t * discount)
                return f_t, g_t

        if self.sde_type == "VE":
            f_t = torch.zeros_like(x_t)
            sigma = (
                self.coef["sigma_min"]
                * (self.coef["sigma_max"] / self.coef["sigma_min"]) ** t
            )
            g_t = sigma * np.sqrt(
                2 * (np.log(self.coef["sigma_max"]) - np.log(self.coef["sigma_min"]))
            )
            return f_t, g_t

    def perturbation_kernel(self, x_0, t: int):
        t = t / self.total_steps  # t in [0, 1]
        if self.sde_type == "VP" or self.sde_type == "subVP":
            log_mean_coeff = (
                -0.25 * t**2 * (self.coef["beta_1"] - self.coef["beta_0"])
                - 0.5 * t * self.coef["beta_0"]
            )
            mean = np.exp(log_mean_coeff) * x_0

            if self.sde_type == "VP":
                std = np.sqrt(1.0 - np.exp(2.0 * log_mean_coeff))
                return mean, std
            if self.sde_type == "subVP":
                std = 1.0 - np.exp(2.0 * log_mean_coeff)
                return mean, std

        if self.sde_type == "VE":
            mean = x_0
            std = (
                self.coef["sigma_min"]
                * (self.coef["sigma_max"] / self.coef["sigma_min"]) ** t
            )
            return mean, std
