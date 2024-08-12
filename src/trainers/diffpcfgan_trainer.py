import typing

import torch
import torch.nn as nn
from PIL import ImageFile

from src.PCF_with_empirical_measure import PCF_with_empirical_measure
from src.utils.utils import cat_linspace_times

ImageFile.LOAD_TRUNCATED_IMAGES = True

from src.trainers.trainer import Trainer


# TODO 12/08/2024 nie_k: Add a way to add a zero at the beginning of a sequence without having to sample it for Swissroll.
# TODO 12/08/2024 nie_k: Alternative plot for swiss roll.


class DiffPCFGANTrainer(Trainer):
    def __init__(
        self,
        generator,
        config,
        learning_rate_gen,
        learning_rate_disc,
        num_D_steps_per_G_step,
        num_samples_pcf,
        hidden_dim_pcf,
        num_diffusion_steps,
        test_metrics_train,
        test_metrics_test,
    ):
        super().__init__(
            test_metrics_train=test_metrics_train,
            test_metrics_test=test_metrics_test,
            num_epochs=config.num_epochs,
        )

        # Parameter for pytorch lightning
        self.automatic_optimization = False

        # Training params
        self.config = config
        self.lr_gen = learning_rate_gen
        self.lr_disc = learning_rate_disc

        # Generator Params
        self.generator = generator

        # Discriminator Params
        self.num_samples_pcf = num_samples_pcf
        self.hidden_dim_pcf = hidden_dim_pcf
        self.discriminator = PCF_with_empirical_measure(
            num_samples=self.num_samples_pcf,
            hidden_size=self.hidden_dim_pcf,
            input_size=self.config.input_dim,
        )
        self.D_steps_per_G_step = num_D_steps_per_G_step

        self.output_dir_images = config.exp_dir

        # Diffusion:
        self.num_diffusion_steps = num_diffusion_steps
        # Initialise the noise parameters
        alphas, betas, baralphas = self.get_noise_level()
        self.alphas = nn.Parameter(alphas, requires_grad=False)
        self.betas = nn.Parameter(betas, requires_grad=False)
        self.baralphas = nn.Parameter(baralphas, requires_grad=False)
        return

    def forward(
        self,
        num_seq: int,
        seq_len: int,
        dim_seq: int,
        noise_seqs_z: typing.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # WIP: explain what noise_seqs_z we need
        if noise_seqs_z is None:
            noise_seqs_z = self._get_noise_vector((num_seq, seq_len, dim_seq))
        else:
            assert noise_seqs_z.shape == (
                num_seq,
                seq_len,
                dim_seq,
            ), f"Expected noise_seqs_z to have shape (num_seq, seq_len, D) but got {noise_seqs_z.shape}"

        # Returns a tensor with shape (num_seq, seq_len, generator.outputdim)
        return self.generator(num_seq, seq_len, self.device, noise_seqs_z)

    def augmented_forward(
        self,
        num_seq: int,
        seq_len: int,
        noise_seqs_z: typing.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Get the starting data: training you start from real data, otherwise sample noise.
        # Use the network for denoising the number of steps. So you need to specify how many steps and you start from full noise.
        # Can we use any network? Generator here useless right?
        # The output is your prediction.
        out = self(
            num_seq=num_seq,
            seq_len=seq_len,
            dim_seq=self.config.input_dim,
            noise_seqs_z=noise_seqs_z,
        )
        out = cat_linspace_times(out)
        return out

    def configure_optimizers(self):
        optim_gen = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.lr_gen,
            weight_decay=0,
            betas=(0, 0.9),
        )
        optim_discr = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr_disc, weight_decay=0
        )
        return [optim_gen, optim_discr], []

    def training_step(self, batch, batch_nb):
        (targets,) = batch
        optim_gen, optim_discr = self.optimizers()

        loss_gen = self._training_step_gen(optim_gen, targets)

        for i in range(self.D_steps_per_G_step):
            self._training_step_disc(optim_discr, targets)

        # Discriminator and Generator share the same loss so no need to report both.
        self.log(
            name="train_pcfd",
            value=loss_gen,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return

    def validation_step(self, batch, batch_nb):
        (targets,) = batch

        diffused_targets: torch.Tensor = self._construct_diffusing_process(
            targets, self._linspace_diffusion_steps
        )
        fake_samples = self.augmented_forward(
            num_seq=targets.shape[0],
            seq_len=targets.shape[1],
            noise_seqs_z=diffused_targets[-1, :, :, :],
        )
        loss_gen = self.discriminator.distance_measure(
            targets, fake_samples, Lambda=0.1
        )

        self.log(
            name="val_pcfd",
            value=loss_gen,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        # TODO 11/08/2024 nie_k: A bit of a hack, I usually code this better but will do the trick for now.
        if not (self.current_epoch + 1) % 50:
            path = (
                self.output_dir_images
                + f"pred_vs_true_epoch_{str(self.current_epoch + 1)}"
            )
            self.evaluate(fake_samples, targets, path)
        return

    def _training_step_gen(self, optim_gen, targets: torch.Tensor) -> float:
        optim_gen.zero_grad()

        diffused_targets: torch.Tensor = self._construct_diffusing_process(
            targets, self._linspace_diffusion_steps
        )

        fake_samples = self.augmented_forward(
            num_seq=targets.shape[0],
            seq_len=targets.shape[1],
            noise_seqs_z=diffused_targets[-1, :, :, :],
        )
        loss_gen = self.discriminator.distance_measure(
            targets, fake_samples, Lambda=0.1
        )

        self.manual_backward(loss_gen)
        optim_gen.step()
        return loss_gen.item()

    def _training_step_disc(self, optim_discr, targets: torch.Tensor) -> float:
        optim_discr.zero_grad()

        diffused_targets: torch.Tensor = self._construct_diffusing_process(
            targets, self._linspace_diffusion_steps
        )

        with torch.no_grad():
            fake_samples = self.augmented_forward(
                num_seq=targets.shape[0],
                seq_len=targets.shape[1],
                noise_seqs_z=diffused_targets[-1, :, :, :],
            )
        loss_disc = -self.discriminator.distance_measure(
            targets, fake_samples, Lambda=0.1
        )
        self.manual_backward(loss_disc)
        optim_discr.step()

        return loss_disc.item()

    def _construct_diffusing_process(
        self, starting_data: torch.Tensor, timestep_diffusion: torch.Tensor
    ) -> torch.Tensor:
        # timestep_diffusion should have values between 0 and diffusion_steps -1, and where 0 corresponds to the initial data.
        # To get the totally noised data, use: output[-1, :, :, :]

        assert timestep_diffusion.dtype in [
            torch.long,
            torch.short,
            torch.bool,
            torch.int,
            torch.int8,
            torch.uint8,
        ], f"timestep_diffusion must be a tensor of type long, byte, or bool but is of type {timestep_diffusion.dtype}"
        assert (
            len(starting_data.shape) == 3
        ), f"Starting data should have shape (N,L,D) but got {starting_data.shape}"
        # For each sequence in L, we compute a final diffused noise value of shape (N,D).
        final_noise_per_sequence = self._get_noise_vector(
            (starting_data.shape[0], 1, starting_data.shape[2])
        )

        # View the data in the format (S,N,L,D) by repeating the data S times.
        starting_data = starting_data.unsqueeze(0).repeat(
            self.num_diffusion_steps + 1, 1, 1, 1
        )

        portion_original_data = torch.pow(self.baralphas[timestep_diffusion], 0.5).view(
            self.num_diffusion_steps + 1, 1, 1, 1
        )
        portion_white_noise = torch.pow(
            1 - self.baralphas[timestep_diffusion], 0.5
        ).view(self.num_diffusion_steps + 1, 1, 1, 1)
        progressively_noisy_data = (
            portion_original_data * starting_data
            + portion_white_noise * final_noise_per_sequence
        )
        # Shape (S, N, L, D). This shape makes sense because we are interested in the tensor N,L,D by slices over S-dim.
        return progressively_noisy_data

    def get_noise_level(self) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Set noising variances betas as in Nichol and Dariwal paper (https://arxiv.org/pdf/2102.09672.pdf)
        s = 0.008

        timesteps = self._linspace_diffusion_steps.float()
        schedule = (
            torch.cos(
                (timesteps / self.num_diffusion_steps + s) / (1.0 + s) * torch.pi / 2.0
            )
            ** 2
        )

        # All following tensors with shape [diffusion_steps]
        baralphas: torch.Tensor = schedule / schedule[0]
        betas: torch.Tensor = 1.0 - baralphas / torch.cat(
            [baralphas[0:1], baralphas[0:-1]]
        )
        alphas: torch.Tensor = 1.0 - betas

        return alphas, betas, baralphas

    @property
    def num_diffusion_steps(self):
        return self._num_diffusion_steps

    @num_diffusion_steps.setter
    def num_diffusion_steps(self, new_num_diffusion_steps):
        if isinstance(new_num_diffusion_steps, int):
            self._num_diffusion_steps = new_num_diffusion_steps
        else:
            raise TypeError(f"num_diffusion_steps is not an {str(int)}.")

    @property
    def _linspace_diffusion_steps(self):
        # One step more because starting point is not diffused.
        return torch.linspace(
            0, self.num_diffusion_steps, self.num_diffusion_steps + 1, dtype=torch.long
        )

    def _get_noise_vector(self, shape: typing.Tuple[int, ...]) -> torch.Tensor:
        return torch.randn(*shape, device=self.device)
