# MNIST

## Data

[Kaggle MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset/data)

60000 training images and 10000 test images
28x28 pixels - each image is a 28x28 matrix of pixels, [0, 255]

## Architecture

LeNet-5
VGG16
AlexNet
GoogLeNet
ResNet

Convulational Neural Network

1. Convolutional Layer (might be multiple)
   hyperparameters:

   - filters/kernels (feature detectors)
     smaller matrices of weights that represent a part of the image (3x3, 5x5, 7x7).
     number of filters influence the depth of the output.
   - padding
     types:
     - valid: no padding
     - same: ensures the output size is the same as the input size.
     - full: padding increases the size of the output by adding zeros to the border of input.
   - stride (step)
     number of pixels the filter moves across the image. A larger stride yields a smaller output.

   first step:

   convolution operation - each filter slides upon height and width of the image performing dot product between filter parameters and image pixels.
   Output of each convolution is a feature map that is represented as a 2D matrix.
   The process is repeated for each filter to generate multiple feature maps.

   second step:

   activation function (non-linearity)

2. Pooling Layer
   reduce dimentionality of the output. Apply filter to the feature map.
   types:

   - max pooling: retains the maximum value in the filter's receptive field.
   - average pooling: averages the values in the filter's receptive field.

3. Fully Connected Layer (Dense Layer)
   flatten the output of the previous layer into a 1D vector.
   choose output dimension (number of neurons)
   activation function to come up with probabilities

   - flattern (from n-dimentional to 1-dimentional)
   - linear transformation (connect input to output)
     output = input × weights + bias
   - activation function (non-linearity)
     ReLU(x) = max(0, x)

## Model

Input processing

Image is represented as a tensor with dimensions [num_images, height, width, channels]
1 channel for gray scale,
3 channels for RGB

files are in hex format.
meta information is in the header:

- 0000: magic number (2051) - uint8 format
- 0002: number of images - 10000
- 0004: number of rows - 28
- 0006: number of columns - 28

##Training

Number of iterations = 10
Batch Size = 32

methods: backpropagation, gradient descent

1. Loss Function - Cross Entropy Loss
2. Optimizer - Adam
3. Accuracy - (correct predictions / total predictions) \* 100

train loop:

- model train mode
- loading to device
- clear gradients
- model prediction
- compute loss
- compute gradients
- backpropagation (updating weights)

test loop:

- model eval mode
- loading to device
- no gradients
- model prediction
- compute loss
- compute accuracy

## Experiments

default config:

- learning rate = 0.001
- batch size = 64
- epochs = 10
- 1 convolutional layer
- 1 fully connected layer

Batch size
Influence:

- frequency of updates (smaller batches -> more frequent updates)
- memory usage (smaller batches -> less memory usage)
- training time (smaller batches -> slower training)
- gradient stability - high gradient variance(larger batches -> more stable)
  Experiments:

1. batch size = 128
2. batch size = 256

Kernel size:
Influence:

- number of parameters (larger kernel -> more parameters 5x5 -> 25 weights)
- pattern recognition (larger kernel -> broader pattern recognition)

experiments:

1. kernel size = 5x5

Stride
Influence:

- output size (larger stride -> smaller output)
- computational efficiency (larger stride -> less computations)
- number of params (larger stride -> less params)
- training time (larger stride -> faster training)

2. stride = 2

Epochs
Influence:

- training time (more epochs -> more training time)

Experiments:

1. epochs = 20
2. epochs = 5

Learning rate
Influence:

- training time (smaller learning rate -> slower training)
- better convergence (smaller learning rate -> more stable convergence)

1. learning rate = 0.0001
2. learning rate = 0.01

Architecture
Influence:

- number of parameters (more layers -> more parameters)
- training time (more layers -> slower training)
- accuracy (more layers -> better accuracy)

Experiments:

1.  2 convolutional layers
    3 fully connected layers

## Weights & Biases

https://api.wandb.ai/links/k-parkhamchuk-kseniya-parkhamchuk/5h5bcgbz

##Inference

- load model and put it into eval mode
- image preprocessing
- model prediction
