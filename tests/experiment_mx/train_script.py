import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.distributions import Normal, Categorical, MixtureSameFamily

from diffusion import DiffusionProcess
from diffusion.utils.sde import SDE
from model import ToyNet

# device = 'mps'
device = "cuda:0"

# Data distribution
mix = Categorical(
    torch.ones(
        2,
    )
)
comp = Normal(torch.tensor([0 - 2.0, 0 + 2.0]), torch.tensor([0.5, 0.5]))
data_dist = MixtureSameFamily(mix, comp)


class SDEProcess(DiffusionProcess):
    def __init__(
        self,
        sde_type="VP",
        sde_info={
            "VP": {"beta_min": 0.1, "beta_max": 20},
            "subVP": {"beta_min": 0.1, "beta_max": 20},
            "VE": {"sigma_min": 0.01, "sigma_max": 50},
        },
        **kwargs
    ):
        super(SDEProcess, self).__init__(**kwargs)
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


score_net = ToyNet(data_dim=1).to(device)
total_steps = 100
num_paths = 32
diffusion = SDEProcess(discrete=False, total_steps=total_steps, sde_type="VP")

x_data = data_dist.sample([num_paths])
x_noise, traj_forward = diffusion.forward_sample(x_data)

plt.figure()
xxx = np.linspace(0, 1, traj_forward.shape[1])
for line in traj_forward:
    plt.plot(xxx, line, linewidth=1.0)
    plt.title("forward")

plt.figure()
x_gen, traj_back = diffusion.sample(torch.randn([num_paths, 1]).to(device), score_net)
xxx = np.linspace(0, 1, traj_forward.shape[1])
for line in traj_back:
    plt.plot(xxx, line.cpu(), linewidth=1.0)
    plt.title("backward")
plt.pause(0.1)

from src.PCF_with_empirical_measure import PCF_with_empirical_measure

pcf = PCF_with_empirical_measure(16, 12, 1)
pcf = pcf.to(device)

num_epochs = 300
opt = torch.optim.Adam(score_net.parameters(), lr=1e-3)
for i in range(num_epochs):
    # Forward process to generate time series
    opt.zero_grad()
    x_data = data_dist.sample([num_paths])
    x_noise, traj_forward = diffusion.forward_sample(x_data)
    traj_forward = traj_forward.to(device)
    # Backward process to generate time series
    x_gen, traj_back = diffusion.backward_sample(
        torch.randn([num_paths, 1]).to(device), score_net
    )
    traj_back_flip = torch.fliplr(traj_back)
    loss = pcf.distance_measure(
        traj_forward.view([-1, total_steps + 1, 1]),
        traj_back_flip.view([-1, total_steps + 1, 1]),
        Lambda=1.0,
    )
    loss.backward()
    opt.step()
    print(i, loss)

x_gen, traj_back = diffusion.sample(torch.randn([num_paths, 1]), score_net)
xxx = np.linspace(0, 1, traj_forward.shape[1])
for line in traj_back:
    plt.plot(xxx, line, linewidth=1.0)
    plt.title("backward")
plt.show()
