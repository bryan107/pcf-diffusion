import logging
import time

import numpy as np
import seaborn as sns
import torch.nn as nn
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.logger.init_logger import set_config_logging
from src.trainers.diffpcfgan_trainer import DiffPCFGANTrainer

set_config_logging()
logger = logging.getLogger(__name__)


from config import ROOT_DIR
from src.networks.lstmgenerator import LSTMGenerator
from src.utils.progressbarwithoutvalbatchupdate import ProgressbarWithoutValBatchUpdate
from src.utils.traininghistorylogger import TrainingHistoryLogger
from src.utils.utils_os import factory_fct_linked_path
from tests.trivial_pcfgan_train.trivialbm_dataset import TrivialBM_Dataset

sns.set()
seed_everything(142, workers=True)

datamodel_name = "pcfgan"
path2file_linker = factory_fct_linked_path(ROOT_DIR, "tests/trivial_pcfgan_train")
datamodel_path = path2file_linker(["out", datamodel_name, ""])
filename_model_saved = "pcfgan_1"

data = TrivialBM_Dataset(1_000)


class Config:
    """Adapter to convert a dictionary to an object with properties/fields."""

    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


config = {
    "device": "cuda",
    "add_time": False,
    "lr_G": 0.01,
    "lr_D": 0.01,
    "D_steps_per_G_step": 2,
    # NUM EPOCHS
    "num_epochs": 501,
    "G_input_dim": 2,
    "G_hidden_dim": 8,
    "input_dim": 2,
    "M_num_samples": 6,
    "M_hidden_dim": 10,
    "lr_M": 0.005,
    "Lambda1": 50,
    "Lambda2": 1,
    "gamma": 0.97,
    # WIP NUM ELEMENT IN SEQ?
    "n_lags": 2,
    "batch_size": 10_000,
    "exp_dir": datamodel_path,
    "gan_algo": "PCFGAN",
    "swa_step_start": 25000,
}
config = Config(config)

period_log: int = 1
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
    filename=filename_model_saved,
)

logger_custom = TrainingHistoryLogger(
    metrics=["train_pcfd", "val_pcfd"],
    aplot_flag=True,
    frequency_epoch_logging=period_log,
)
epochs = 1001

trainer = Trainer(
    default_root_dir=path2file_linker(["out"]),
    # gradient_clip_val=0.1,
    gpus=1,
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
lstm_generator = LSTMGenerator(
    input_dim=2,
    hidden_dim=8,
    output_dim=1,
    n_layers=1,
    noise_scale=1.0,
    BM=True,
    activation=nn.Identity(),
)
model = (
    # PCFGANTrainer(
    DiffPCFGANTrainer(
        generator=lstm_generator,
        config=config,
        learning_rate_gen=config.lr_G,
        learning_rate_disc=config.lr_D,
        num_D_steps_per_G_step=config.D_steps_per_G_step,
        num_samples_pcf=config.M_num_samples,
        hidden_dim_pcf=config.M_hidden_dim,
        num_diffusion_steps=3,
        # wip: THESE TWO SEEM UNUSED??
        test_metrics_train=None,
        test_metrics_test=None,
        # WIP I THINK BATCH SIZE DOES SMTHG DIFFERENT
    )
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
plt.show()
