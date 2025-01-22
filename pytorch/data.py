import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Custom Dataset class
class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = self.transform(image)

        return image, label


# data preprocessing
transform = transforms.Compose(
    [
        # transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Lambda(lambda img: torch.unsqueeze(img, dim=0)),
        # 0.1307 is a mean of MNIST dataset (sum/ pixel_count)
        # 0.3081 standard deviation
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)


# reading from image file
def read_image_file(image_file):
    with open(image_file, "rb") as f:  # read binary
        data_format = int.from_bytes(
            f.read(4), "big"
        )  # should be 2051 (MNIST magic number uint8 format)

        if data_format != 2051:
            raise ValueError(
                f"Invalid MNIST image file, expected magic numbe 2051, got {data_format}"
            )
        num_images = int.from_bytes(
            f.read(4), "big"
        )  # reads next 4 bytes big endian byte-order, cursor shifted by 4
        rows = int.from_bytes(f.read(4), "big")  # cursor shifted by 4
        cols = int.from_bytes(f.read(4), "big")  # cursor shifted by 4

        images = torch.frombuffer(f.read(), dtype=torch.uint8).view(
            num_images, rows, cols
        )  # creates an array of num_images size with each item of 28x28 values representing pixels

        return images  # converts numpy array to torch tensor


def read_image_labels(image_labels_file):
    with open(image_labels_file, "rb") as f:
        data_format = int.from_bytes(f.read(4), "big")  # should be 2049

        if data_format != 2049:
            raise ValueError(
                f"Invalid MNIST label file, expect magic number 2049, got {data_format}"
            )

        f.seek(8)  # skip 8 header bytes
        labels = torch.frombuffer(
            f.read(), dtype=torch.uint8
        )  # return tensor with labels
        print(labels)
        return labels


# Creating Datasets and Dataloaders


def load_mnist_train_data():
    labels = read_image_labels("data/train/train-labels.idx1-ubyte")
    images = read_image_file("data/train/train-images.idx3-ubyte")
    return images, labels


def load_mnist_test_data():
    labels = read_image_labels("data/test/t10k-labels.idx1-ubyte")
    images = read_image_file("data/test/t10k-images.idx3-ubyte")
    return images, labels


def get_mnist_dataloader(dataset="train", batch_size=32, shuffle=True):
    if dataset == "train":
        images, labels = load_mnist_train_data()
    elif dataset == "test":
        images, labels = load_mnist_test_data()
    else:
        raise ValueError("dataset must be either 'train' or 'test'")

    mnist_dataset = MNISTDataset(images, labels, transform=transform)
    return DataLoader(mnist_dataset, batch_size, shuffle=True)


# Visualization


def visualize_MNIST_item():
    train_features, train_labels = next(iter(get_mnist_dataloader("train")))
    print(f"Feature batch size: {train_features.size()}")
    print(f"Labels batch size: {train_labels.size()}")

    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")  # prepare for showing img, creates a plot
    plt.show()  # actually showing img
    print(f"Label: {label}")
