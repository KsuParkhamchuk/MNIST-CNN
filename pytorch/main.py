from train import train_model
from hyperparameters import *
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():

    print(f"Configuration: epochs={EPOCHS}, batch_size={BATCH_SIZE}, learning_rate={LEARNING_RATE}")

    try:
        logger.debug('Starting model training')
        train_model(epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.exception("Detailed traceback:")
        raise

if __name__ == '__main__':
    main()