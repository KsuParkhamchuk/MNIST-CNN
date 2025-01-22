# Ouput shape: (batch_size, number_of_filters, height, width)

import torch
from torch import nn


class Convolution2d:
    # input is a batch of images of (channel, w, h)
    def __init__(self, input, kernel_size, padding, stride, in_channels, out_channels):
        self.input = input
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = self.init_kernel()

    def forward(self):
        batch_size, *_ = self.input.shape
        output_dim_x, output_dim_y = self.get_output_dimentions()

        output = torch.zeros(
            (batch_size, self.out_channels, output_dim_x, output_dim_y)
        )

        for b in range(batch_size):
            windows = self.get_all_windows(window_size=self.kernel_size)
            for out_ch in range(self.out_channels):
                for i, window in enumerate(windows):
                    # Calculate x, y coordinates from the window index
                    y = i // output_dim_x
                    x = i % output_dim_x
                    output[b, out_ch, y, x] = self.convolve(window=window[b], ch=out_ch)
        return output

    # kernel has 4d shape: out_filters x in_filters x width x height
    def init_kernel(self):
        # total number of input connections
        # if fan_in increases we need to reduce weights
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        kaiming_init = torch.sqrt(torch.tensor(2.0 / fan_in))
        # (mean, deviation, size)
        return nn.Parameter(
            torch.normal(
                mean=0.0,
                std=kaiming_init,
                size=(
                    self.out_channels,
                    self.in_channels,
                    self.kernel_size,
                    self.kernel_size,
                ),
            )
        )

    def get_all_windows(self, window_size):
        windows = []

        # add padding to input
        padded_input = torch.nn.functional.pad(
            self.input, (self.padding, self.padding, self.padding, self.padding)
        )

        output_width, output_height = self.get_output_dimentions()

        # Extract windows considering stride and all channels
        for i in range(0, output_height * self.stride, self.stride):
            for j in range(0, output_width * self.stride, self.stride):
                window = padded_input[:, :, i : i + window_size, j : j + window_size]
                windows.append(window)

        return windows

    # window shape is (batch_size, in_channels, kernel_size, kernel_size)
    def convolve(self, window, ch):
        height, width = self.kernel_size, self.kernel_size

        output = 0.0
        for c in range(self.in_channels):
            for i in range(height):
                for j in range(width):
                    output += window[c, i, j] * self.kernel[ch, c, i, j]

        return output

    def get_output_dimentions(self):
        _, _, height, width = self.input

        output_dim_x = int(
            (width + 2 * self.padding - self.kernel_size) / self.stride + 1
        )
        output_dim_y = int(
            (height + 2 * self.padding - self.kernel_size) / self.stride + 1
        )

        return output_dim_x, output_dim_y
