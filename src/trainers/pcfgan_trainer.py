import numpy as np
import torch
from PIL import ImageFile
from tqdm import tqdm

from src.PCF_with_empirical_measure import PCF_with_empirical_measure

ImageFile.LOAD_TRUNCATED_IMAGES = True

from src.trainers.trainer import Trainer


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
        self.discriminator = PCF_with_empirical_measure(
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
