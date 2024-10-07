"""
Running this script will save a trained model at
datamodel_path
and save images of the training inside.

It erases the folder before running the script.

"""

import logging
import signal
import sys
import time

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from src.logger.init_logger import set_config_logging

set_config_logging()
logger = logging.getLogger(__name__)

from src.trainers.visual_data import DataType
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from src.networks.models.toynet import ToyNet
from config import ROOT_DIR
from src.trainers.diffpcfgan_trainer import DiffPCFGANTrainer
from src.utils.progressbarwithoutvalbatchupdate import ProgressbarWithoutValBatchUpdate
from src.utils.traininghistorylogger import TrainingHistoryLogger
from src.utils.utils_os import factory_fct_linked_path, remove_files_from_dir, savefig
from tests.trivial_pcfgan_train.trivialbm_dataset import TrivialBM_Dataset

sns.set()
seed_everything(142, workers=True)

datamodel_name = "pcfgan"
path2file_linker = factory_fct_linked_path(ROOT_DIR, "tests/trivial_pcfgan_train")
datamodel_path = path2file_linker(["out", datamodel_name, ""])

lr_gen = 0.0018
lr_disc = 0.001
num_samples_pcf = 10
hidden_dim_pcf = 8
num_diffusion_steps = 8
parser_type = "Truncation"
parser_len = 8
scheduler_type = "Step"
use_fixed_measure_discriminator_pcfd = True

########## Delete the previous run if it exists
remove_files_from_dir(datamodel_path)
###############################################

data = TrivialBM_Dataset(500, 5_000)

PERIOD_LOG: int = 5
PERIOD_IN_LOG_PLOTTING: int = 40
early_stop_val_loss = EarlyStopping(
    monitor="val_epdf",
    min_delta=1e-4,
    patience=4000 // PERIOD_LOG,
    verbose=True,
    mode="min",
)
chkpt = ModelCheckpoint(
    monitor="val_epdf",
    mode="min",
    verbose=True,
    save_top_k=1,
    dirpath=datamodel_path,
    filename="model",
)

logger_custom = TrainingHistoryLogger(
    metrics=[
        "train_score_matching",
        "val_score_matching",
        "train_pcfd",
        "val_pcfd",
        "train_epdf",
        "val_epdf",
    ],
    plot_loss_history=True,
    period_logging_pt_lightning=PERIOD_LOG,
    period_in_logs_plotting=PERIOD_IN_LOG_PLOTTING,
)
EPOCHS = 20_001

trainer = Trainer(
    default_root_dir=path2file_linker(["out"]),
    gpus=[0],
    max_epochs=EPOCHS,
    logger=[logger_custom],
    check_val_every_n_epoch=PERIOD_LOG,
    num_sanity_val_steps=0,
    callbacks=[
        early_stop_val_loss,
        ProgressbarWithoutValBatchUpdate(refresh_rate=10),
        chkpt,
    ],
)

logger.info("Creating the model.")
score_network = ToyNet(data_dim=data.inputs.shape[1])
model = DiffPCFGANTrainer(
    data_train=data.train_in,
    data_val=data.val_in,
    score_network=score_network,
    learning_rate_gen=lr_gen,
    learning_rate_disc=lr_disc,
    num_D_steps_per_G_step=1,
    num_samples_pcf=num_samples_pcf,
    hidden_dim_pcf=hidden_dim_pcf,
    num_diffusion_steps=num_diffusion_steps,
    data_type=DataType.ONE_D,
    output_dir_images=datamodel_path,
    parser_type=parser_type,
    parser_len=parser_len,
    sched_type=scheduler_type,
    use_fixed_measure_discriminator_pcfd=use_fixed_measure_discriminator_pcfd,
)
logger.info("Model created.")

# section ######################################################################
#  #############################################################################
#  Training


### Catch errors etc:
def terminating_operations():
    savefig(logger_custom.fig, datamodel_path + f"loss_history.png")
    return


# Signal handler to save the results if the program is interrupted.
def termination_handler(signum, frame):
    logger.critical(f"Signal {signum} received, saving data and exiting...")
    try:
        terminating_operations()
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    sys.exit(0)


# When the script is stopped softly (not using -9), we will wrap up the work.
signal.signal(signal.SIGINT, termination_handler)
signal.signal(signal.SIGTERM, termination_handler)

try:
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
except Exception as e:
    logger.error(f"Training failed: {e}")
    raise e
finally:
    terminating_operations()
plt.show()
