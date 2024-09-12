"""
Running this script will save a trained model at
datamodel_path
and save images of the training inside.

It erases the folder before running the script.

"""

import logging
import time

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.logger.init_logger import set_config_logging

set_config_logging()
logger = logging.getLogger(__name__)


from src.networks.models.toynet import ToyNet
from config import ROOT_DIR
from src.trainers.diffpcfgan_trainer import DiffPCFGANTrainer
from src.utils.progressbarwithoutvalbatchupdate import ProgressbarWithoutValBatchUpdate
from src.utils.traininghistorylogger import TrainingHistoryLogger
from src.utils.utils_os import factory_fct_linked_path, savefig, remove_files_from_dir
from tests.trivial_pcfgan_train.trivialbm_dataset import TrivialBM_Dataset

sns.set()
seed_everything(142, workers=True)

datamodel_name = "pcfgan_disc_long_diff_pcfd_loss"
path2file_linker = factory_fct_linked_path(ROOT_DIR, "tests/trivial_pcfgan_train")
datamodel_path = path2file_linker(["out", datamodel_name, ""])


########## Delete the previous run if it exists
remove_files_from_dir(datamodel_path)
###############################################

data = TrivialBM_Dataset(750, 5_000)


class Config:
    """Adapter to convert a dictionary to an object with properties/fields."""

    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


config = {
    "device": "cuda",
    "lr_G": 0.0005,
    "lr_D": 0.001,
    "D_steps_per_G_step": 1,
    "G_input_dim": 2,
    "input_dim": data.inputs.shape[2],
    "M_num_samples": 16,
    "M_hidden_dim": 12,
    # WIP NUM ELEMENT IN SEQ?
    "n_lags": data.inputs.shape[1],
    "exp_dir": datamodel_path,
}
config = Config(config)

period_log: int = 5
period_in_logs_plotting: int = 40
early_stop_val_loss = EarlyStopping(
    monitor="train_pcfd",
    min_delta=1e-4,
    patience=2000 // period_log,
    verbose=True,
    mode="min",
)
chkpt = ModelCheckpoint(
    monitor="train_pcfd",
    mode="min",
    verbose=True,
    save_top_k=1,
    dirpath=datamodel_path,
    filename="model",
)

logger_custom = TrainingHistoryLogger(
    metrics=[
        "train_pcfd",
        "val_pcfd",
        "train_score_matching",
        "val_score_matching",
        "train_reconst",
        "val_reconst",
        "train_epdf",
        "val_epdf",
    ],
    plot_loss_history=True,
    period_logging_pt_lightning=period_log,
    period_in_logs_plotting=period_in_logs_plotting,
)
epochs = 5001

trainer = Trainer(
    default_root_dir=path2file_linker(["out"]),
    # gradient_clip_val=0.1,
    gpus=[3],
    max_epochs=epochs,
    logger=[logger_custom],
    check_val_every_n_epoch=period_log,
    num_sanity_val_steps=0,
    callbacks=[
        early_stop_val_loss,
        ProgressbarWithoutValBatchUpdate(refresh_rate=10),
        chkpt,
    ],
)

logger.info("Creating the model.")
score_network = ToyNet(data_dim=config.input_dim)
model = DiffPCFGANTrainer(
    data_train=data.train_in,
    data_val=data.val_in,
    score_network=score_network,
    config=config,
    learning_rate_gen=config.lr_G,
    learning_rate_disc=config.lr_D,
    num_D_steps_per_G_step=config.D_steps_per_G_step,
    num_samples_pcf=config.M_num_samples,
    hidden_dim_pcf=config.M_hidden_dim,
    num_diffusion_steps=32,
    use_fixed_measure_discriminator_pcfd=False,
)
logger.info("Model created.")

# section ######################################################################
#  #############################################################################
#  Training

start_time = time.perf_counter()
trainer.fit(model, datamodule=data)
train_time = np.round(time.perf_counter() - start_time, 2)
print(
    "Total time training: ",
    train_time,
    " seconds. In average, it took: ",
    np.round(train_time / trainer.current_epoch, 4),
    " seconds per epochs.",
)

savefig(logger_custom.fig, config.exp_dir + f"loss_history.png")
plt.show()
