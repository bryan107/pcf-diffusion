import os
from abc import ABC, abstractmethod, ABCMeta


class DiffusionMeta(ABCMeta):
    """
    Metaclass for DiffusionProcess to ensure the abstract methods are enforced.
    """


class DiffusionProcess(ABC, metaclass=DiffusionMeta):
    def __init__(self, discrete: bool, total_steps: int, schedule: str = "cosine"):
        self.discrete = discrete
        self.total_steps = total_steps
        self.schedule = schedule
        self.result_dir = os.path.join(".", "results")

        # Ensure the results directory exists
        os.makedirs(self.result_dir, exist_ok=True)

    ### Misnamed or wrong?
    # def compute_coef(self):
    #     """
    #     Compute the coefficients based on the schedule and total steps.
    #     """
    #     self.coef = compute_ddpm_coef(self.total_steps, self.schedule)

    @abstractmethod
    def forward_one_step(self, x_prev, t):
        """
        Abstract method to perform one forward diffusion step.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def forward_steps(self, x_0, t, noise):
        """
        Abstract method to perform forward diffusion across multiple steps.
        Must be implemented by subclasses.
        """
        pass

    def plot_forward_steps(self, x_0):
        """
        Plot the forward diffusion process from x_0 to x_T.
        Saves the intermediate results as images and creates a GIF.
        """

        data_type = "toy" if x_0.dim() == 2 else "img"
        initial_image_path = os.path.join(self.result_dir, "x_0.png")
        noise_image_path = os.path.join(self.result_dir, "noise.png")

        # Save the initial state and a noise image
        save2img(x_0, initial_image_path, data_type, self.discrete)
        save2img(torch.randn_like(x_0), noise_image_path, data_type, self.discrete)

        trajectory = [initial_image_path]

        x_t = x_0
        for t in range(1, self.total_steps + 1):
            x_t = self.forward_one_step(x_t, t)
            if x_t.ndim > 2:
                x_t.clamp_(-1.0, 1.0)

            step_image_path = os.path.join(self.result_dir, f"x_{t}.png")
            save2img(x_t, step_image_path, data_type, self.discrete)
            trajectory.append(step_image_path)

        # Create a GIF from the saved images
        make_gif(trajectory, "forward_process")

        # Clean up the intermediate image files
        for path in trajectory:
            os.remove(path)
