from model import MNISTCNN
from data import get_mnist_dataloader
import torch
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

torch.set_printoptions(threshold=float("inf"), linewidth=1000)


class Inference:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.model.eval()

    def load_model(self):
        model = MNISTCNN()
        checkpoint = torch.load(
            "checkpoint.pth", map_location=self.device, weights_only=True
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        return model

    def predict(self):
        test_dataloader = get_mnist_dataloader("test")
        data_loader_iter = iter(test_dataloader)

        try:
            logger.debug("Start inference model")
            with torch.no_grad():
                images, labels = next(data_loader_iter)
                images = images.to(self.device)
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)  # 32x10
                prediction = torch.argmax(
                    probabilities, dim=1
                )  # max out of each dimention from probabilities
                return prediction
        except Exception as e:
            logger.error(e)
            logger.exception("Detailed traceback:")


def main():
    inference = Inference()
    inference.predict()


if __name__ == "__main__":
    main()
