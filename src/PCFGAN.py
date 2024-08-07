import numpy as np
import torch
from PIL import ImageFile
from torch import nn
from tqdm import tqdm

from src.pathdevelopment.unitarydevelopmentlayer import UnitaryDevelopmentLayer
from src.trainers.trainer import Trainer

ImageFile.LOAD_TRUNCATED_IMAGES = True


class char_func_path(nn.Module):
    def __init__(
        self,
        num_samples,
        hidden_size,
        input_size,
        init_range: float = 1,
    ):
        """
        Class for computing path charateristic function.

        Args:
            num_samples (int): Number of samples.
            hidden_size (int): Hidden size.
            input_size (int): Input size.
            add_time (bool): Whether to add time dimension to the input.
            init_range (float, optional): Range for weight initialization. Defaults to 1.
        """
        super().__init__()
        self.num_samples = num_samples
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.input_size = input_size
        self.unitary_development = UnitaryDevelopmentLayer(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            channels=self.num_samples,
            include_inital=True,
            return_sequence=False,
            init_range=init_range,
        )
        for param in self.unitary_development.parameters():
            param.requires_grad = True

    @staticmethod
    def HS_norm(X: torch.tensor, Y: torch.Tensor):
        """
        Hilbert-Schmidt norm computation.

        Args:
            X (torch.Tensor): Complex-valued tensor of shape (C, m, m).
            Y (torch.Tensor): Tensor of the same shape as X.

        Returns:
            torch.float: Hilbert-Schmidt norm of X and Y.
        """
        if len(X.shape) == 4:
            m = X.shape[-1]
            X = X.reshape(-1, m, m)

        else:
            pass
        D = torch.bmm(X, torch.conj(Y).permute(0, 2, 1))
        return (torch.einsum("bii->b", D)).mean().real

    def distance_measure(
        self, X1: torch.tensor, X2: torch.tensor, Lambda=0.1
    ) -> torch.float:
        """
        TODO: this description is just not true.
        Distance measure given by the Hilbert-Schmidt inner product.

        Args:
            X1 (torch.tensor): Time series samples with shape (N_1, T, d).
            X2 (torch.tensor): Time series samples with shape (N_2, T, d).
            Lambda (float, optional): Scaling factor for additional distance measure on the initial time point,
            this is found helpful for learning distribution of initial time point.
              Defaults to 0.1.

        Returns:
            torch.float: Distance measure between two batches of samples.
        """

        ########## DEPRECATED
        # print(X1.shape)
        # if self.add_time:
        #     X1 = AddTime(X1)
        #     X2 = AddTime(X2)
        # else:
        #     pass
        # print(X1.shape)
        #########################
        dev1, dev2 = self.unitary_development(X1), self.unitary_development(X2)
        N, T, d = X1.shape

        # initial_dev = self.unitary_development_initial()
        CF1, CF2 = dev1.mean(0), dev2.mean(0)

        if Lambda != 0:
            initial_incre_X1 = torch.cat(
                [torch.zeros((N, 1, d)).to(X1.device), X1[:, 0, :].unsqueeze(1)], dim=1
            )
            initial_incre_X2 = torch.cat(
                [torch.zeros((N, 1, d)).to(X1.device), X2[:, 0, :].unsqueeze(1)], dim=1
            )
            initial_CF_1 = self.unitary_development(initial_incre_X1).mean(0)
            initial_CF_2 = self.unitary_development(initial_incre_X2).mean(0)
            return self.HS_norm(CF1 - CF2, CF1 - CF2) + Lambda * self.HS_norm(
                initial_CF_1 - initial_CF_2, initial_CF_1 - initial_CF_2
            )
        else:
            return self.HS_norm(CF1 - CF2, CF1 - CF2)


class PCFGANTrainer(Trainer):
    def __init__(
        self,
        generator,
        train_dataset,
        config,
        test_metrics_train,
        test_metrics_test,
    ):
        """
        Trainer class for the basic PCF-GAN, without time serier embedding module.

        Args:
            generator (torch.nn.Module): PCFG generator model.
            train_dl (torch.utils.data.DataLoader): Training data loader.
            config: Configuration object containing hyperparameters and settings.
            **kwargs: Additional keyword arguments for the base trainer class.
        """
        super().__init__(
            test_metrics_train=test_metrics_train,
            test_metrics_test=test_metrics_test,
            num_epochs=config.num_epochs,
        )

        # Training params
        self.config = config
        self.batch_size = config.batch_size
        char_input_dim = self.config.input_dim

        # Generator Params
        self.generator = generator
        self.generator_optim = torch.optim.Adam(
            generator.parameters(), lr=config.lr_G, betas=(0, 0.9), weight_decay=0
        )
        self.G_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.generator_optim, gamma=config.gamma
        )

        # Discriminator Params
        self.discriminator = char_func_path(
            num_samples=config.M_num_samples,
            hidden_size=config.M_hidden_dim,
            input_size=char_input_dim,
        )
        self.discriminator_optim = torch.optim.Adam(
            self.discriminator.parameters(), lr=config.lr_M
        )
        self.D_steps_per_G_step = config.D_steps_per_G_step

        # Dataset
        self.train_dl = train_dataset
        return

    def fit(self, device):
        """
        Trains the PCFGAN model.

        Args:
            device: Device to perform training on.
        """

        self.generator.to(device)
        self.discriminator.to(device)

        for i in tqdm(range(self.num_epochs)):
            self.step(device, i)
            if i > self.config.swa_step_start:
                self.averaged_G.update_parameters(self.generator)

    def step(self, device, step):
        """
        Performs one training step.

        Args:
            device: Device to perform training on.
            step (int): Current training step.
        """
        D_losses_this_epoch = []
        targets = next(iter(self.train_dl))[0].to(device)

        # Discriminator training
        for i in range(self.D_steps_per_G_step):
            D_loss = self._training_step_discr(targets, device)
            D_losses_this_epoch.append(D_loss)
        self.losses_history["D_loss"].append(np.mean(D_losses_this_epoch))

        # Generator training
        G_loss = self._training_step_gen(targets, device)
        self.losses_history["G_loss"].append(G_loss)

        if step % 100 == 0:
            fake_samples = self.generator(
                batch_size=self.batch_size,
                n_lags=self.config.n_lags,
                device=device,
            )
            self.evaluate(fake_samples, targets, step, self.config)

        if step % 500 == 0:
            self.G_lr_scheduler.step()
            for param_group in self.generator_optim.param_groups:
                print("Learning Rate: {}".format(param_group["lr"]))

        return

    def _training_step_gen(self, targets, device):
        """
        Performs one training step for the generator.

        Args:
            targets: Real samples for training.
            device: Device to perform training on.

        Returns:
            float: Generator loss value.
        """
        fake_samples = self.generator(
            batch_size=self.batch_size,
            n_lags=self.config.n_lags,
            device=device,
        )
        self.generator.train()
        self.generator_optim.zero_grad()
        G_loss = self.discriminator.distance_measure(targets, fake_samples, Lambda=0.1)
        G_loss.backward()
        self.generator_optim.step()
        return G_loss.item()

    def _training_step_discr(self, targets, device):
        """
        Performs one training step for the discriminator.

        Args:
            targets: Real samples for training.
            device: Device to perform training on.
        Returns:
            float: Discriminator loss value.
        """
        self.discriminator_optim.zero_grad()

        with torch.no_grad():
            # TODO 07/08/2024 nie_k: looks sus to ask [0] but next iter is common for dataloaders.
            fake_samples = self.generator(
                batch_size=self.batch_size,
                n_lags=self.config.n_lags,
                device=device,
            )
        d_loss = -self.discriminator.distance_measure(targets, fake_samples, Lambda=0.1)
        d_loss.backward()

        self.discriminator_optim.step()
        return d_loss.item()
