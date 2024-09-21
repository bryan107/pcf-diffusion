import torch
from diffusion import DiffusionProcess
from diffusion.utils.sde import SDE
import numpy as np
from matplotlib import pyplot as plt

class SDEProcess(DiffusionProcess):
    def __init__(self, sde_type="VP",
                 sde_info={"VP": {"beta_min": 0.1, "beta_max": 20},
                           "subVP": {"beta_min": 0.1, "beta_max": 20},
                           "VE": {"sigma_min": 0.01, "sigma_max": 50}}, 
                device = 'cuda:0',
                 **kwargs):
        super(SDEProcess, self).__init__(**kwargs)
        assert self.discrete is False, "DDPM is only for continuous data"
        self.dt = 1. / self.total_steps # step size
        self.sde = SDE(self.total_steps, sde_type, sde_info)
        self.device = device


    def forward_one_step(self, x_prev, t):
        """
        Discretized forward SDE process for actual compuatation: 
        x_{t+1} = x_t + f_t(x_t) * dt + G_t * z_t * sqrt(dt)
        """
        f_t, g_t = self.sde.drifts(x_prev, t-1)
        z = torch.randn_like(x_prev)
        x_t = x_prev + f_t * self.dt + g_t * z * np.sqrt(self.dt)
        return x_t

    
    def backward_one_step(self, x_t, t, pred_score, clip_denoised=True):
        """
        Discretized backward SDE process for actual compuatation:
        x_{t-1} = x_t - (f_t(x_t) - (G_t)^2 * pred_score) * dt + G_t * z_t * sqrt(dt)
        """
        #x_t = x_t.detach()
        z = torch.randn_like(x_t).to(self.device)
        f_t, g_t = self.sde.drifts(x_t, t)
        f_t = f_t.to(self.device)
        #print(f_t, g_t, z)
        x_prev = x_t - (f_t - g_t**2 * pred_score) * self.dt + g_t * z * np.sqrt(self.dt)
        if clip_denoised and x_t.ndim > 2:
            print('backward_one_step')
            x_prev.clamp_(-1., 1.)

        return x_prev

    @torch.no_grad()
    def sample(self, noise, net):
        """
        Sample from backward diffusion process
        """
        x_t = noise
        trajs = [x_t]

        for t in reversed(range(1, self.total_steps+1)):
            pred_score = net(x_t, t)
            x_t = self.backward_one_step(x_t, t, pred_score)
            trajs.append(x_t)
        return x_t, torch.stack(trajs, dim=2)
    
    def forward_sample(self, data):
        trajs = torch.zeros([data.shape[0], data.shape[1], self.total_steps+1])
        x = data.to(self.device)
        trajs[:, :, 0] = x
        for t in range(1, self.total_steps+1):
            x = self.forward_one_step(x, t)
            trajs[:, :, t] = x
        return x, trajs
    
    def backward_sample(self, noise, net):
        """
        Sample from backward diffusion process
        """
        x_t = noise
        trajs = [x_t]

        for t in reversed(range(1, self.total_steps+1)):
            pred_score = net(x_t, t)
            x_t = self.backward_one_step(x_t, t, pred_score)
            trajs.append(x_t)
        return x_t, torch.stack(trajs, dim=2)
    
def plot_trajs(forward_trajs, backward_trajs, ylim=[-5, 5]):
    xxx = np.linspace(0, 1, forward_trajs.shape[2])
    time_steps = forward_trajs.shape[2]
    forward_trajs = forward_trajs.view(-1, time_steps)
    backward_trajs = backward_trajs.view(-1, time_steps)
    #print(forward_trajs.shape, backward_trajs.shape)
    plt.figure(figsize=(12,3))

    plt.subplot(1, 2, 1)
    for line in forward_trajs:
        #print(line.shape)
        plt.plot(xxx, line, linewidth=1.0)
        plt.title('forward')
    plt.ylim(-5, 5)
    
    plt.subplot(1, 2, 2)
    for line in backward_trajs:
        plt.plot(xxx, line.cpu(), linewidth=1.0)
        plt.title('backward')
    plt.ylim(-5, 5)

    plt.show()  