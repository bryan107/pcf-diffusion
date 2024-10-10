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
# Put below set_config_logging to get detail logs about early stopping.
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.logger.init_logger import set_config_logging

set_config_logging()
logger = logging.getLogger(__name__)

from tests.parameters_product import parameters_product
from src.trainers.visual_data import DataType
from pytorch_lightning import seed_everything, Trainer
from tests.swiss_roll_pcfgan_train.swissroll_dataset import SwissRoll_Dataset
from src.networks.models.toynet import ToyNet
from config import ROOT_DIR
from src.trainers.diffpcfgan_trainer import DiffPCFGANTrainer
from src.utils.progressbarwithoutvalbatchupdate import ProgressbarWithoutValBatchUpdate
from src.utils.traininghistorylogger import TrainingHistoryLogger
from src.utils.utils_os import factory_fct_linked_path, remove_files_from_dir, savefig

sns.set()

########## All parameters to modify for training:
#####################################################
PARAMS_GRID = {
    "lr_gen": [0.000_3],
    "lr_disc": [0.000_1],
    "num_samples_pcf": [12],
    "hidden_dim_pcf": [8],
    "num_diffusion_steps": [32, 128],
    "parser_type": ["Truncation", "Subsampling"],
    "parser_len": [16, 32],
    "scheduler_type": ["Step", "Cosine"],
    "use_fixed_measure_discriminator_pcfd": [False],
}
###### Other constants:
#####################################################
PERIOD_LOG: int = 5
PERIOD_IN_LOG_PLOTTING: int = 40
PATIENCE = 4000
EPOCHS = 20_001
GPU_ID = [4]
SEED = 142
DATA_SIZE = 1000


#####################################################
#####################################################
def get_dir_name_from_params(
    lr_gen,
    lr_disc,
    num_diffusion_steps,
    parser_type,
    parser_len,
    scheduler_type,
    use_fixed_measure_discriminator_pcfd,
):
    dir_name = (
        f"pcfgan"
        f"_diff{num_diffusion_steps}"
        f"_pars{parser_type[:4].lower()}"
        f"_len{parser_len}"
        f"_sched{scheduler_type[:4].lower()}"
        f"_lrgen{format(lr_gen, '.5g').replace('.', ',')}"  # Replace dot with comma in the learning rate
        f"_fixed{str(use_fixed_measure_discriminator_pcfd).lower()}"
    )
    return dir_name


def run_training(config, verbose):
    # Extract parameters from config
    lr_gen = config["lr_gen"]
    lr_disc = config["lr_disc"]
    num_samples_pcf = config["num_samples_pcf"]
    hidden_dim_pcf = config["hidden_dim_pcf"]
    num_diffusion_steps = config["num_diffusion_steps"]
    parser_type = config["parser_type"]
    parser_len = config["parser_len"]
    scheduler_type = config["scheduler_type"]
    use_fixed_measure_discriminator_pcfd = config[
        "use_fixed_measure_discriminator_pcfd"
    ]

    seed_everything(SEED, workers=True)
    datamodel_name = get_dir_name_from_params(
        lr_gen,
        lr_disc,
        num_diffusion_steps,
        parser_type,
        parser_len,
        scheduler_type,
        use_fixed_measure_discriminator_pcfd,
    )
    path2file_linker = factory_fct_linked_path(
        ROOT_DIR, "tests/swiss_roll_pcfgan_train"
    )
    datamodel_path = path2file_linker(["out", datamodel_name, ""])
    ########## Delete the previous run if it exists
    remove_files_from_dir(datamodel_path)
    ###############################################
    data = SwissRoll_Dataset(DATA_SIZE, True)

    early_stop_val_loss = EarlyStopping(
        monitor="val_epdf",
        min_delta=1e-4,
        patience=PATIENCE // PERIOD_LOG,
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

    trainer = Trainer(
        default_root_dir=path2file_linker(["out"]),
        gpus=GPU_ID,
        max_epochs=EPOCHS,
        logger=[logger_custom],
        check_val_every_n_epoch=PERIOD_LOG,
        num_sanity_val_steps=0,
        callbacks=[
            early_stop_val_loss,
            ProgressbarWithoutValBatchUpdate(refresh_rate=10 if verbose else 0),
            chkpt,
        ],
    )
    if verbose:
        logger.info("Creating the model.")
    score_network = ToyNet(data_dim=data.inputs.shape[2])
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
        data_type=DataType.TWO_D,
        output_dir_images=datamodel_path,
        parser_type=parser_type,
        parser_len=parser_len,
        sched_type=scheduler_type,
        use_fixed_measure_discriminator_pcfd=use_fixed_measure_discriminator_pcfd,
    )
    if verbose:
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
        logger.info(
            f"Total time training: {train_time} seconds. On average, it took: {np.round(train_time / trainer.current_epoch, 4)} seconds per epoch."
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise e
    finally:
        terminating_operations()


if __name__ == "__main__":
    # Main loop to iterate over parameter combinations
    configs = parameters_product(PARAMS_GRID)
    logger.info(f"Number of configurations: {len(configs)}")
    for i, config in enumerate(configs):
        logger.info("================================================")
        logger.info(f"================= CONFIG {i}/{len(configs)} ===================")
        logger.info("================================================")
        logger.info(f"Running training with configuration: {config}")
        try:
            run_training(config, verbose=(len(configs) < 2))
        except Exception as e:
            logger.error(f"Training with config {config} \nfailed for the reason: {e}.")
            logger.error("Continuing with the next configuration.")

logger.info("All configurations have been trained. Hope it was successful :)")
plt.show()
