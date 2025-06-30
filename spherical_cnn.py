# gemini-2.5 first pass code, under review
"""
spherical_cnn.py

This file defines the architecture of a simple Spherical Convolutional Neural Network.
It uses the `s2fft` library to perform the spherical harmonic transforms, which
are the core of spherical convolutions.

The model consists of:
- A 2D convolutional layer to embed the input mask into a higher-dimensional space.
- A sequence of three custom SphericalConv2d layers.
- A final 2D convolutional layer to unembed the features back to a single-channel S-SDF.
"""
import torch
import torch.nn as nn
import s2fft

# Monkey-patching for older s2fft versions if necessary
# s2fft.signal.spherical_conv = s2fft.spherical_conv

class SphericalConv2d(nn.Module):
    """
    A Spherical Convolutional Layer.

    This layer performs a convolution on the sphere by:
    1. Transforming the input signal and the kernel to the harmonic domain using s2fft.
    2. Performing element-wise multiplication in the harmonic domain.
    3. Transforming the result back to the spatial domain.
    """
    def __init__(self, in_channels, out_channels, L, sampling="mw"):
        super(SphericalConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.L = L  # Spherical harmonic bandlimit
        self.sampling = sampling

        # Learnable weights for the convolution kernel
        # The kernel size is determined by the number of channels and the bandlimit L
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, L, 2*L-1, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, dtype=torch.float32))

    def forward(self, x):
        # x shape: (batch, in_channels, L, 2L-1)
        batch_size, in_channels, L, _ = x.shape
        
        # Pre-transform weights (ideally in __init__)
        weight_lm = torch.zeros(self.out_channels, self.in_channels, self.L, 2*self.L-1, dtype=torch.complex64, device=x.device)
        for o in range(self.out_channels):
            for i in range(self.in_channels):
                weight_lm[o, i, :, :] = torch.from_numpy(s2fft.forward(self.weight[o, i, :, :].cpu().numpy(), L=self.L, sampling=self.sampling)).to(x.device)

        # Initialize output tensor
        # Initialize output tensor
        out = torch.zeros(batch_size, self.out_channels, self.L, 2*self.L-1, device=x.device)

        # Loop over batch 
        for b in range(batch_size):
            # Transform input channels
            x_lm_sample = torch.zeros(in_channels, self.L, 2*self.L-1, dtype=torch.complex64, device=x.device)
            for i in range(in_channels):
                x_lm_sample[i, :, :] = torch.from_numpy(s2fft.forward(x[b, i, :, :].cpu().numpy(), L=self.L, sampling=self.sampling)).to(x.device)

            # Convolve
            out_lm_sample = torch.einsum('icl,oicl->ocl', x_lm_sample, weight_lm)

            # Inverse transform output channels
            for o in range(self.out_channels):
                out[b, o, :, :] = torch.from_numpy(s2fft.inverse(out_lm_sample[o, :, :].cpu().numpy(), L=self.L, sampling=self.sampling)).to(x.device)

        # Add bias
        out = out + self.bias
        
        return out

class SimpleSphericalCNN(nn.Module):
    """
    A simple Spherical CNN for the S-SDF generation task.
    """
    def __init__(self, L, intermediate_channels=16):
        super(SimpleSphericalCNN, self).__init__()
        self.L = L

        # 1. Embedder: A standard 2D convolution to lift the input mask to more channels
        self.embedder = nn.Conv2d(
            in_channels=1, 
            out_channels=intermediate_channels, 
            kernel_size=3, 
            padding=1 # Padding to keep the size the same
        )

        # 2. Spherical Convolutional Layers
        self.sph_conv1 = SphericalConv2d(intermediate_channels, intermediate_channels, L)
        self.sph_conv2 = SphericalConv2d(intermediate_channels, intermediate_channels, L)
        self.sph_conv3 = SphericalConv2d(intermediate_channels, intermediate_channels, L)
        
        # Activation function
        self.activation = nn.ReLU()

        # 3. Unembedder: A standard 2D convolution to project back to a single channel
        self.unembedder = nn.Conv2d(
            in_channels=intermediate_channels, 
            out_channels=1, 
            kernel_size=3, 
            padding=1
        )

    def forward(self, x):
        # x shape: (batch, 1, L, L)
        
        # Embed
        x = self.embedder(x)
        x = self.activation(x)
        
        # Spherical Convolutions
        x = self.sph_conv1(x)
        x = self.activation(x)
        
        x = self.sph_conv2(x)
        x = self.activation(x)
        
        x = self.sph_conv3(x)
        x = self.activation(x)
        
        # Unembed
        x = self.unembedder(x)
        
        return x
