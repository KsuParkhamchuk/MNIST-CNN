import wandb
from hyperparameters import *

def init_wandb():

    wandb.init(
        project="MNIST",
        name=f"run_batch={BATCH_SIZE}_lr={LEARNING_RATE}_epochs={EPOCHS}_filter={KERNEL_SIZE}",

        # Hyperparams
        config={
        "learning_rate": LEARNING_RATE,
        "architecture": ARCHITECHTURE,
        "dataset": DATASET,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "kernel_size": KERNEL_SIZE,
        "stride": STRIDE
        }
    )

    return wandb

def log_metrics(metrics):
    wandb.log(metrics)

def finish_run():
    wandb.finish()