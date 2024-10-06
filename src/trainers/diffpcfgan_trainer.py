import logging
import math
import typing

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import MultiStepLR

from src.metrics.epdf import HistogramLoss
from src.trainers.visual_data import DataType
from src.utils.utils_os import savefig

logger = logging.getLogger(__name__)

from src.diffusionsequenceparser import (
    DiffusionSequenceParser,
    SubsamplingParser, TruncationParser,
)
from src.pcfempiricalmeasure import PCFEmpiricalMeasure
from src.differentialequations.diffusionprocess_continuous import (
    SDEType,
    ContinuousDiffusionProcess,
)

# For the method: plot_for_back_ward_trajectories
sns.set()

PERIOD_PLOT_VAL = 100

### WIP SIMUL_VARIABLES CHANGE THESE STEPS, DO 8, 32 and 64
NUM_STEPS_DIFFUSION_2_CONSIDER = 8
# Adding 1 for the zero at the beginning.
NUM_STEPS_DIFFUSION_2_CONSIDER += 1

# Type annotation for a model that takes two tensors (x_t, time_step) and returns a tensor
ScoreNetworkType = typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class DiffPCFGANTrainer(LightningModule):

    @staticmethod
    def get_noise_vector(shape: typing.Tuple[int, ...], device: str) -> torch.Tensor:
        return torch.randn(*shape, device=device)

    @staticmethod
    def _flat_add_time_transpose_and_add_zero(data: torch.Tensor) -> torch.Tensor:
        # Receive data of shape (S, N, L, D).
        # Transforms it into (N, S, L * D)

        # Merge the sequence dimension (dim 2) and the feature dimension (dim 3) into a single dimension.
        # Flattening is required before adding time otherwise we would add too many dimensions with the time.
        data = data.flatten(2, 3)

        # Add a zero at the beginning of the sequence. By adding it before time simplifies time augmentation.
        # WIP MOVE THIS
        zeros = torch.zeros(1, data.shape[1], data.shape[2], device=data.device)
        data = torch.cat((zeros, data), dim=0)

        # Add the times to the data by creating a linspace between 0,1 and which is repeated for all (N,L).
        diffusion_times = (
            torch.linspace(
                0.0,
                1.0,
                steps=data.shape[0],
                device=data.device,
            )
            .view(-1, 1, 1)
            .expand(
                data.shape[0],
                data.shape[1],
                1,
            )
        )
        data = torch.cat((data, diffusion_times), dim=-1)
        # Permute the batch axis and the diffusion axis.
        data = data.transpose(0, 1)
        # adding contiguous slows down the code tremendously, so we keep it like that.
        return data

    @staticmethod
    def flattenNtranspose(data: torch.Tensor) -> torch.Tensor:
        # Receive data of shape (S, N, L, D).
        # Return data of shape (N, S, L * D)
        return data.flatten(2, 3).transpose(0, 1)

    @staticmethod
    def from_paths_to_loss(
        diffused_targets: torch.Tensor,
        denoised_diffused_targets: torch.Tensor,
        step_training_for_logs: typing.Optional[str],
        sampling_parser: typing.Optional[DiffusionSequenceParser] = None,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            diffused_targets (torch.Tensor): The diffused target sequences of shape (S, N, L, D).
            denoised_diffused_targets (torch.Tensor): The denoised diffused target sequences of shape (S, N, L, D).
            step_training_for_logs (Optional[str]): A string indicating the current training step for logging purposes. If not given, do not log.
            sampling_parser (Optional[DiffusionSequenceParser]): A callable function that
                processes the target sequences. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        #### Prepare for PCFD loss the diffusion trajectories.
        diffused_targets = diffused_targets[:-1]
        denoised_diffused_targets = denoised_diffused_targets[:-1]

        # Remove last value that is identical.
        diffused_targets4pcfd = DiffPCFGANTrainer._flat_add_time_transpose_and_add_zero(
            diffused_targets
        )
        denoised_diffused_targets4pcfd = (
            DiffPCFGANTrainer._flat_add_time_transpose_and_add_zero(
                denoised_diffused_targets
            )
        )
        if step_training_for_logs is not None:
            logger.debug(
                f"\nDiffused targets for {step_training_for_logs}: \n%s\nDenoised samples for {step_training_for_logs}: %s\n",
                diffused_targets4pcfd,
                denoised_diffused_targets4pcfd,
            )

        if sampling_parser is not None:
            diffused_targets4pcfd = sampling_parser(diffused_targets4pcfd)
            denoised_diffused_targets4pcfd = sampling_parser(
                denoised_diffused_targets4pcfd
            )

            if step_training_for_logs is not None:
                logger.debug(
                    f"\nDiffused targets for {step_training_for_logs} after parsing: \n%s\nDenoised samples for {step_training_for_logs}: %s\n",
                    diffused_targets4pcfd,
                    denoised_diffused_targets4pcfd,
                )

        # Sequences are of shape (S, N, L, D). We transform into the correct format (N, S, L * D).
        diffused_targets = DiffPCFGANTrainer.flattenNtranspose(diffused_targets)
        denoised_diffused_targets = DiffPCFGANTrainer.flattenNtranspose(
            denoised_diffused_targets
        )

        return (
            diffused_targets,
            denoised_diffused_targets,
            diffused_targets4pcfd,
            denoised_diffused_targets4pcfd,
        )

    def __init__(
        self,
        data_train: torch.Tensor,
        data_val: torch.Tensor,
        score_network: ScoreNetworkType,
        config: dict,
        learning_rate_gen: float,
        learning_rate_disc: float,
        num_D_steps_per_G_step: int,
        num_samples_pcf: int,
        hidden_dim_pcf: int,
        num_diffusion_steps: int,
        data_type: DataType,
        use_fixed_measure_discriminator_pcfd: bool = False,
    ):
        # score_network is used to denoise the data and will be called as score_net(data, time).

        super().__init__()

        self.data_type: DataType = data_type

        # Parameter for pytorch lightning
        self.automatic_optimization: bool = False

        # Training params
        self.config = config
        self.lr_gen: float = learning_rate_gen
        self.lr_disc: float = learning_rate_disc

        # Score Network Params
        self.score_network: ScoreNetworkType = score_network

        # Discriminator Params
        self.num_samples_pcf: int = num_samples_pcf
        self.hidden_dim_pcf: int = hidden_dim_pcf
        self.discriminator = PCFEmpiricalMeasure(
            num_samples=self.num_samples_pcf,
            hidden_size=self.hidden_dim_pcf,
            # TODO 13/08/2024 nie_k: instead of input_dim, set time_series_for_compar_dim
            input_size=self.config.input_dim * self.config.n_lags + 1,
        )
        self.D_steps_per_G_step: int = num_D_steps_per_G_step
        self.use_fixed_measure_discriminator_pcfd = use_fixed_measure_discriminator_pcfd

        self.output_dir_images: str = config.exp_dir

        # Diffusion:
        self.diffusion_process = ContinuousDiffusionProcess(
            total_steps=num_diffusion_steps,
            schedule="cosine",
            sde_type=SDEType.VP,
        )
        self.num_diffusion_steps: int = num_diffusion_steps

        ####### WIP CHANGE THIS SIMUL_VARIABLES
        if True:
            self.sampling_parser: typing.Optional[DiffusionSequenceParser] = (
                TruncationParser(NUM_STEPS_DIFFUSION_2_CONSIDER)
            )
        else:
            self.sampling_parser: typing.Optional[DiffusionSequenceParser] = (
                SubsamplingParser(NUM_STEPS_DIFFUSION_2_CONSIDER)
            )

        ####
        # WIP to explain:
        # we pass to the histo loss the data that we are interested in. This
        # might be different from the data we input. Hence, there should
        # be a mechanism that goes from data input to data interested in.
        # On the generation side, we create these data points that we are interested in.
        # In that regard, we know the number of features;
        if (
            self.data_type is DataType.ONE_D
            or self.data_type is DataType.TWO_D
            or self.data_type is DataType.THREE_D
        ):
            self.num_axes_per_samples = 1
        else:
            self.num_axes_per_samples = data_train.shape[-1]
        ### Loses
        self.use_diffusion_score_matching_loss = False
        self.L2_loss = torch.nn.MSELoss()
        # Instantiate the HistogramLoss
        self.train_histo_loss = HistogramLoss(
            data_train, int(round(2.0 * math.pow(data_train.shape[0], 1.0 / 3.0), 0))
        )
        self.val_histo_loss = HistogramLoss(
            data_val, int(round(2.0 * math.pow(data_val.shape[0], 1.0 / 3.0), 0))
        )

        # Initialize the axes for plotting the trajectories (backward and forward, 2 axes).
        self.plot_diffusion_fig, self.plot_diffusion_axes = plt.subplots(
            1, 2, sharey=True
        )
        # Initialize the axes for plotting the samples. One axe per feature.
        self.plot_samples_fig, self.plot_samples_axes = plt.subplots(
            1, self.num_axes_per_samples
        )
        # Just make sure it is a list.
        if self.num_axes_per_samples == 1:
            self.plot_samples_axes = [self.plot_samples_axes]

        # Initialize the axes for plotting the evolution of the diffusion samples
        NUM_PLOT_EVOLUTION_DIFF = 5
        self.plot_evol_fig, self.plot_evol_axes = plt.subplots(
            self.num_axes_per_samples, NUM_PLOT_EVOLUTION_DIFF, figsize=[12.6, 4.8]
        )
        # Reshape into a matrix format which is easy to handle for plots.
        self.plot_evol_axes = self.plot_evol_axes.reshape(
            self.num_axes_per_samples, NUM_PLOT_EVOLUTION_DIFF
        )
        return

    def get_backward_path(
        self,
        *,
        num_seq: typing.Optional[int] = None,
        seq_len: typing.Optional[int] = None,
        dim_seq: typing.Optional[int] = None,
        noise_start_seq_z: typing.Optional[torch.Tensor] = None,
        proba_teacher_forcing: float = 0.0,
        # Don't reverse, handled inside. The order should be start with original data and finishes with noise.
        # Shape (S,N,L,D).
        teacher_forcing_inputs=None,
    ) -> torch.Tensor:
        # Alias for forward for clarity
        return self(
            num_seq=num_seq,
            seq_len=seq_len,
            dim_seq=dim_seq,
            noise_start_seq_z=noise_start_seq_z,
            proba_teacher_forcing=proba_teacher_forcing,
            teacher_forcing_inputs=teacher_forcing_inputs,
        )

    def forward(
        self,
        *,
        num_seq: typing.Optional[int] = None,
        seq_len: typing.Optional[int] = None,
        dim_seq: typing.Optional[int] = None,
        noise_start_seq_z: typing.Optional[torch.Tensor] = None,
        proba_teacher_forcing: float = 0.0,
        teacher_forcing_inputs=None,
    ) -> torch.Tensor:
        # Denoise data to generate new samples.
        # Along the first dimension, the first value corresponds to the output data (generated samples).
        assert (
            num_seq is not None and seq_len is not None and dim_seq is not None
        ) or noise_start_seq_z is not None, "Either the three parameters num_seq, seq_len, dim_seq need to be given or the noise_start_seq_z."

        # WIP: explain what noise_start_seq_z we need
        if noise_start_seq_z is None:
            noise_start_seq_z = DiffPCFGANTrainer.get_noise_vector(
                (num_seq, seq_len, dim_seq), self.device
            )

        traj_back = self.diffusion_process.backward_sample(
            noise_start_seq_z,
            self.score_network,
            proba_teacher_forcing=proba_teacher_forcing,
            sequences_forcing=teacher_forcing_inputs,
        )

        # Returns a tensor with shape (num_step_diffusion, num_seq, seq_len, generator.outputdim).
        # Along the first dimension, the first value corresponds to the output data (generated samples).
        return traj_back.flip(0)

    def configure_optimizers(self):
        optim_gen = torch.optim.Adam(
            self.score_network.parameters(),
            lr=self.lr_gen,
            weight_decay=0,
            betas=(0, 0.9),
        )
        optim_discr = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr_disc, weight_decay=0
        )

        # A good rule of thumb is that if the max_epoch is 10k, patience is 3k, then you need to reduce the learning rate
        # at least every 3k epoch.
        schedul_optim_gen = MultiStepLR(
            optim_gen,
            milestones=[
                2 * self.trainer.max_epochs // 5,
                4 * self.trainer.max_epochs // 5,
            ],
            gamma=0.1,
        )
        return [optim_gen, optim_discr], [schedul_optim_gen]

    def training_step(self, batch, batch_nb):
        targets = batch[0]
        optim_gen, optim_discr = self.optimizers()

        logger.debug("Targets for training: %s", targets)
        losses_as_dict = self._training_step_gen(optim_gen, targets)

        if not self.use_fixed_measure_discriminator_pcfd:
            for i in range(self.D_steps_per_G_step):
                _ = self._training_step_disc(optim_discr, targets)

        # Discriminator and Generator share the same loss so no need to report both.
        self._log_all_metrics(
            {
                "pcfd": losses_as_dict["train_pcfd"],
                "score_matching": losses_as_dict["train_score_matching"],
                "epdf": losses_as_dict["train_epdf"],
            },
            "train_",
        )
        return

    def validation_step(self, batch, batch_nb):
        targets = batch[0]

        logger.debug("Targets for validation: %s", targets)
        diffused_targets: torch.Tensor = self._get_forward_path(targets, [])
        denoised_diffused_targets: torch.Tensor = self.get_backward_path(
            noise_start_seq_z=diffused_targets[-1],
            proba_teacher_forcing=0.0,
        )

        (
            diffused_targets,
            denoised_diffused_targets,
            diffused_targets4pcfd,
            denoised_diffused_targets4pcfd,
        ) = self.from_paths_to_loss(
            diffused_targets,
            denoised_diffused_targets,
            "validation",
            self.sampling_parser,
        )

        loss_gen = self.discriminator.distance_measure(
            diffused_targets4pcfd,
            denoised_diffused_targets4pcfd,
            lambda_y=0.0,
        )

        loss_gen_score_matching = 0.0
        if self.use_diffusion_score_matching_loss:
            loss_gen_score_matching = self._compute_score_matching_loss(targets)
        loss_gen_epdf = self.val_histo_loss(denoised_diffused_targets[:, :1])

        self._log_all_metrics(
            {
                "pcfd": loss_gen,
                "score_matching": loss_gen_score_matching,
                "epdf": loss_gen_epdf,
            },
            "val_",
        )

        # TODO 11/08/2024 nie_k: A bit of a hack, I usually code this better but will do the trick for now.
        # TODO 29/08/2024 nie_k: The plot need to be change depending on dataset (manually) and also would not work for sequences
        if not (self.current_epoch + 1) % PERIOD_PLOT_VAL:
            # Clear axes before plotting
            self._clear_axes()

            self.evaluate(denoised_diffused_targets, diffused_targets)

            self.plot_for_back_ward_trajectories(
                denoised_diffused_targets, diffused_targets
            )
            plt.pause(0.01)
        return

    def evaluate(self, backward_trajectory, forward_trajectory):
        path_img_saved_prediction = (
            self.output_dir_images + f"pred_vs_true_epoch_{str(self.current_epoch + 1)}"
        )

        path_img_saved_diffusion_evol = (
            self.output_dir_images + f"diff_evol_epoch_{str(self.current_epoch + 1)}"
        )

        if self.data_type.plot_method:
            # Call the plotting function
            self.data_type.plot_method(
                forward_trajectory[:, 0],
                backward_trajectory[:, 0],
                self.plot_samples_axes,
                path_img_saved_prediction,
                True,
            )
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")

        # Generate a list of evenly spaced indices across the diffusion steps
        steps_for_plot = np.linspace(
            0, self.num_diffusion_steps - 1, self.plot_evol_axes.shape[1], dtype=int
        )

        for i, step in enumerate(steps_for_plot):
            self.data_type.plot_method(
                forward_trajectory[:, step],
                backward_trajectory[:, step],
                self.plot_evol_axes[:, i],
                None,
                False,
            )
            self.plot_evol_axes[0, i].set_title(
                f"Step {step}/{self.num_diffusion_steps - 1}"
            )
        self.plot_evol_fig.tight_layout()
        savefig(self.plot_evol_fig, path_img_saved_diffusion_evol)
        return

    def plot_for_back_ward_trajectories(
        self, denoised_diffused_targets, diffused_targets
    ):
        assert (
            denoised_diffused_targets.shape == diffused_targets.shape
        ), f"Shapes are not the same: {denoised_diffused_targets.shape} and {diffused_targets.shape}."
        assert (
            len(denoised_diffused_targets.shape) == 3
        ), f"Expected 3 dimensions but got {len(denoised_diffused_targets.shape)} dimensions."

        denoised_diffused_targets = denoised_diffused_targets.detach().cpu().numpy()
        diffused_targets = diffused_targets.detach().cpu().numpy()

        diffusion_steps = np.arange(denoised_diffused_targets.shape[1])

        ### Forward
        for element_dataset in range(diffused_targets.shape[0]):
            self.plot_diffusion_axes[0].plot(
                diffusion_steps,
                diffused_targets[element_dataset, :, 0],
                linewidth=1.0,
            )
        self.plot_diffusion_axes[0].set_title("Forward Path")
        self.plot_diffusion_axes[0].set_xlabel("Diffusion Step")

        ### Backward
        for element_dataset in range(denoised_diffused_targets.shape[0]):
            self.plot_diffusion_axes[1].plot(
                diffusion_steps,
                denoised_diffused_targets[element_dataset, :, 0],
                linewidth=1.0,
            )
        self.plot_diffusion_axes[1].invert_xaxis()  # Automatically reverse the x-axis

        self.plot_diffusion_axes[1].set_title("Backward Path")
        self.plot_diffusion_axes[1].set_xlabel("Diffusion Step")

        # Set figure title and layout
        self.plot_diffusion_fig.suptitle(
            f"Comparison Diffusion Trajectories for n={diffused_targets.shape[0]}. \n"
            f"The distributions are matched over the first {NUM_STEPS_DIFFUSION_2_CONSIDER} steps."
        )
        self.plot_diffusion_fig.tight_layout()

        savefig(
            self.plot_diffusion_fig,
            self.output_dir_images + f"trajectories_{str(self.current_epoch + 1)}.png",
        )
        return

    @property
    def proba_teacher_forcing(self):
        """
        During the first half of the training epochs, the probability smoothly decreases following
        a cosine schedule. In the second half, the probability is fixed at zero.

        - The cosine schedule is used for epochs in the range [0, max_epochs // 2).
        - After reaching half of the maximum number of epochs, the probability is set to 0.

        """
        if False:
            return 0.5 * (
                1
                + torch.cos(
                    torch.tensor(
                        self.current_epoch * math.pi / (self.trainer.max_epochs // 2)
                    )
                )
            )
        else:
            return torch.tensor([0.0])

    def _training_step_gen(
        self, optim_gen, targets: torch.Tensor
    ) -> typing.Dict[str, float]:
        optim_gen.zero_grad()

        diffused_targets: torch.Tensor = self._get_forward_path(targets, [])
        denoised_diffused_targets: torch.Tensor = self.get_backward_path(
            noise_start_seq_z=diffused_targets[-1],
            proba_teacher_forcing=self.proba_teacher_forcing,
            teacher_forcing_inputs=diffused_targets,
        )

        (
            diffused_targets,
            denoised_diffused_targets,
            diffused_targets4pcfd,
            denoised_diffused_targets4pcfd,
        ) = self.from_paths_to_loss(
            diffused_targets,
            denoised_diffused_targets,
            "training",
            self.sampling_parser,
        )

        loss_gen = self.discriminator.distance_measure(
            diffused_targets4pcfd,
            denoised_diffused_targets4pcfd,
            lambda_y=0.0,
        )
        total_loss = loss_gen

        loss_gen_score_matching = 0.0
        if self.use_diffusion_score_matching_loss:
            loss_gen_score_matching = self._compute_score_matching_loss(targets)
            total_loss = total_loss + 0.1 * loss_gen_score_matching

        self.manual_backward(total_loss)
        optim_gen.step()

        loss_gen_epdf = self.train_histo_loss(denoised_diffused_targets[:, :1])
        return {
            "train_pcfd": loss_gen,
            "train_score_matching": loss_gen_score_matching,
            "train_epdf": loss_gen_epdf,
        }

    def _training_step_disc(
        self, optim_discr, targets: torch.Tensor
    ) -> typing.Dict[str, float]:
        optim_discr.zero_grad()

        with torch.no_grad():
            diffused_targets: torch.Tensor = self._get_forward_path(targets, [])
            denoised_diffused_targets: torch.Tensor = self.get_backward_path(
                noise_start_seq_z=diffused_targets[-1],
                proba_teacher_forcing=self.proba_teacher_forcing,
                teacher_forcing_inputs=diffused_targets,
            )

            (
                _,
                _,
                diffused_targets4pcfd,
                denoised_diffused_targets4pcfd,
            ) = self.from_paths_to_loss(
                diffused_targets, denoised_diffused_targets, None, self.sampling_parser
            )

        loss_disc = -self.discriminator.distance_measure(
            diffused_targets4pcfd,
            denoised_diffused_targets4pcfd,
            lambda_y=0.0,
        )
        self.manual_backward(loss_disc)
        optim_discr.step()

        return {
            "train_pcfd": loss_disc,
        }

    def _get_forward_path(
        self,
        starting_data: torch.Tensor,
        # Ignore features to be diffused with a hack here.
        indices_features_not_diffuse: typing.Iterable = [-1],
    ) -> torch.Tensor:
        # To get the totally noised data, use: output[-1, :, :, :]

        assert (
            len(starting_data.shape) == 3
        ), f"Incorrect shape for starting_data: Expected 3 dimensions (N, L, D) but got {len(starting_data.shape)} dimensions with shape {starting_data.shape}. Make sure the tensor is correctly reshaped or initialized."

        diffused_starting_data: torch.Tensor = (
            starting_data.clone()
            .unsqueeze(0)
            .repeat(self.num_diffusion_steps + 1, 1, 1, 1)
        )

        mask_where_diffuse = torch.ones(starting_data.shape[-1], dtype=torch.bool)
        mask_where_diffuse[indices_features_not_diffuse] = False

        diffused_starting_data[:, :, :, mask_where_diffuse] = (
            self.diffusion_process.forward_sample(
                starting_data[:, :, mask_where_diffuse]
            )
        )

        # Shape (S, N, L, D). This shape makes sense because we are interested in the tensor N,L,D by slices over S-dim.
        return diffused_starting_data

    def _compute_score_matching_loss(self, targets):
        time_step_diffusion = torch.randint(
            1, self.num_diffusion_steps + 1, (1,), device=self.device
        )
        _, diffusion = self.diffusion_process._compute_drift_and_diffusion(
            torch.zeros_like(targets), time_step_diffusion
        )
        mean, std = self.diffusion_process._perturbation_kernel(
            targets, time_step_diffusion
        )
        noise = torch.randn_like(targets)
        perturbed_noise = mean + std * noise
        pred_score = self.score_network(perturbed_noise, time_step_diffusion)
        # NCSN score matching objective function (x_tilda - x) / sigma^2
        target = -noise / std
        loss_gen_score_matching = (
            diffusion * diffusion * self.L2_loss(pred_score, target)
        )
        return loss_gen_score_matching

    def _log_all_metrics(self, metrics: typing.Dict[str, float], prefix: str):
        # For convenience, we have a method here that will log all metrics appropriately. The only thing to pass is
        # the prefix to the name of the metric, essentially "train_" or "val_".
        for name, value in metrics.items():
            self.log(
                name=prefix + name,
                value=value,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
        return

    def _clear_axes(self):
        for ax in self.plot_diffusion_axes:
            ax.clear()
        for ax in self.plot_samples_axes:
            ax.clear()
        for ax_list in self.plot_evol_axes:
            for ax in ax_list:
                ax.clear()
        return
