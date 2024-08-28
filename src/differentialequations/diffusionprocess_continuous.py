from enum import Enum
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class SDEType(Enum):
    """Enum representing the types of Stochastic Differential Equations (SDEs)."""

    VP = "VP"  # Variance Preserving
    SUB_VP = "subVP"  # Sub Variance Preserving
    VE = "VE"  # Variance Exploding


class ContinuousDiffusionProcess(nn.Module):
    def __init__(
        self,
        total_steps: int,
        # WIP I don't think this is used.
        schedule: str,
        sde_type: SDEType = SDEType.VP,
        sde_info: Dict[SDEType, Dict[str, float]] = {
            "VP": {"beta_min": 0.1, "beta_max": 20},
            "subVP": {"beta_min": 0.1, "beta_max": 20},
            "VE": {"sigma_min": 0.01, "sigma_max": 50},
        },
    ):
        """
        Initialize the ContinuousDiffusionProcess.

        Args:
            total_steps (int): The total number of steps in the diffusion process.
            schedule (str): The schedule for the diffusion process.
            sde_type (SDEType): The type of SDE to use (VP, subVP, VE).
            sde_info (Dict[SDEType, Dict[str, float]]): A dictionary containing parameters for each SDE type.
        """
        super().__init__()

        self.total_steps = total_steps
        self.schedule = schedule
        if sde_info is None:
            sde_info = {
                SDEType.VP: {"beta_min": 0.1, "beta_max": 20},
                SDEType.SUB_VP: {"beta_min": 0.1, "beta_max": 20},
                SDEType.VE: {"sigma_min": 0.01, "sigma_max": 50},
            }

        self.dt: float = 1.0 / self.total_steps  # Step size for SDE discretization
        self.sde_type: SDEType = sde_type
        self.sde_params: Dict[str, float] = sde_info[sde_type]
        self.coefficients: Dict[str, float] = self._compute_coefficients()

    def forward_sample(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate forward samples using the SDE.

        Args:
            data (torch.Tensor): The input data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The final sample and the trajectory.
        """
        x = data
        trajectory = [x]

        for t in range(1, self.total_steps + 1):
            x = self._forward_one_step(x, t)
            trajectory.append(x)

        # Use torch.stack for the trajectory
        return x, torch.stack(trajectory, dim=1)

    def backward_sample(
        self, noise: torch.Tensor, model: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the diffusion process using the backward SDE.

        Args:
            noise (torch.Tensor): The noise final point of the equation, starting the backward equation.
            model (nn.Module): The model to predict the score.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The final sample and the trajectory.
        """
        x_t = noise
        trajectory = [x_t]

        for t in reversed(range(1, self.total_steps + 1)):
            pred_score = model(x_t, t)
            x_t = self._backward_one_step(x_t, t, pred_score)
            trajectory.append(x_t)

        return x_t, torch.stack(trajectory, dim=1)

    def _forward_one_step(self, x_prev: torch.Tensor, t: int) -> torch.Tensor:
        """
        Perform one forward step of the SDE.

        Args:
            x_prev (torch.Tensor): The state at the previous timestep.
            t (int): The current timestep.

        Returns:
            torch.Tensor: The state at the next timestep.
        """
        drift, diffusion = self._compute_drift_and_diffusion(x_prev, t - 1)
        noise = torch.randn_like(x_prev)
        x_t = x_prev + drift * self.dt + diffusion * noise * np.sqrt(self.dt)
        return x_t

    def _backward_one_step(
        self,
        x_t: torch.Tensor,
        t: int,
        pred_score: torch.Tensor,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """
        Perform one backward step of the SDE.

        Args:
            x_t (torch.Tensor): The state at the current timestep.
            t (int): The current timestep.
            pred_score (torch.Tensor): The predicted score from the model.
            clip_denoised (bool): Whether to clip the denoised output.

        Returns:
            torch.Tensor: The state at the previous timestep.
        """
        drift, diffusion = self._compute_drift_and_diffusion(x_t, t)
        noise = torch.randn_like(x_t, device=x_t.device)
        x_prev = (
            x_t
            - (drift - diffusion**2 * pred_score) * self.dt
            + diffusion * noise * np.sqrt(self.dt)
        )

        if clip_denoised and x_t.ndim > 2:
            x_prev = x_prev.clamp(-1.0, 1.0)

        return x_prev

    def _compute_coefficients(self) -> Dict[str, float]:
        """
        Compute the coefficients for the SDE based on the selected type.

        Returns:
            Dict[str, float]: The coefficients for the SDE process.
        """
        if self.sde_type in {SDEType.VP, SDEType.SUB_VP}:
            return {
                "beta_0": self.sde_params["beta_min"],
                "beta_1": self.sde_params["beta_max"],
            }

        if self.sde_type == SDEType.VE:
            return {
                "sigma_min": self.sde_params["sigma_min"],
                "sigma_max": self.sde_params["sigma_max"],
            }

    def _compute_drift_and_diffusion(
        self, diffused_process: torch.Tensor, t: int
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute the drift and diffusion coefficients for the SDE at a given timestep.

        Args:
            diffused_process (torch.Tensor): The state at time t of shape (batch_size, *).
            t (int): The current timestep as an index.

        Returns:
            Tuple[torch.Tensor, float]: The drift (f_t) and diffusion (g_t) coefficients.
        """
        normalized_t = t / self.total_steps  # Normalize timestep to [0, 1]

        if self.sde_type in {SDEType.VP, SDEType.SUB_VP}:
            beta_t = self.coefficients["beta_0"] + normalized_t * (
                self.coefficients["beta_1"] - self.coefficients["beta_0"]
            )
            drift = -0.5 * beta_t * diffused_process

            if self.sde_type == SDEType.VP:
                diffusion = np.sqrt(beta_t)
            elif self.sde_type == SDEType.SUB_VP:
                discount = 1.0 - np.exp(
                    -2.0 * self.coefficients["beta_0"] * normalized_t
                    - (self.coefficients["beta_1"] - self.coefficients["beta_0"])
                    * normalized_t**2.0
                )
                diffusion = np.sqrt(beta_t * discount)

            return drift, diffusion

        elif self.sde_type == SDEType.VE:
            drift = torch.zeros_like(diffused_process)
            sigma_t = (
                self.coefficients["sigma_min"]
                * (self.coefficients["sigma_max"] / self.coefficients["sigma_min"])
                ** normalized_t
            )
            diffusion = sigma_t * np.sqrt(
                2.0
                * (
                    np.log(self.coefficients["sigma_max"])
                    - np.log(self.coefficients["sigma_min"])
                )
            )
            return drift, diffusion

    def _perturbation_kernel(
        self, x_0: torch.Tensor, t: int
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, float]]:
        """
        Compute the perturbation kernel for the SDE.

        Args:
            x_0 (torch.Tensor): The initial state at time 0.
            t (int): The current timestep as index.

        Returns:
            Tuple[torch.Tensor, Union[torch.Tensor, float]]: The mean and standard deviation for the perturbation kernel.
        """
        normalized_t = t / self.total_steps  # Normalize timestep to [0, 1]

        if self.sde_type in {SDEType.VP, SDEType.SUB_VP}:
            log_mean_coeff = (
                -0.25
                * normalized_t**2.0
                * (self.coefficients["beta_1"] - self.coefficients["beta_0"])
                - 0.5 * normalized_t * self.coefficients["beta_0"]
            )
            mean = np.exp(log_mean_coeff) * x_0

            if self.sde_type == SDEType.VP:
                std = np.sqrt(1.0 - np.exp(2.0 * log_mean_coeff))
            elif self.sde_type == SDEType.SUB_VP:
                std = 1.0 - np.exp(2.0 * log_mean_coeff)

            return mean, std

        elif self.sde_type == SDEType.VE:
            mean = x_0
            std = (
                self.coefficients["sigma_min"]
                * (self.coefficients["sigma_max"] / self.coefficients["sigma_min"])
                ** normalized_t
            )
            return mean, std
