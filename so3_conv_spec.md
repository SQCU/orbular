## API Contract: SphericalConvBlock

This document specifies the API for a `SphericalConvBlock`, a module that implements the core components of the scaled Spherical CNN.

### 1. Overview

The `SphericalConvBlock` takes a set of spherical feature maps as input and produces a new set of spherical feature maps as output. It encapsulates a spherical convolution, batch normalization, and a non-linear activation function. It can optionally include a residual connection and perform spectral pooling.

### 2. Input Tensor Specification

The module must accept an input tensor with the following characteristics:

*   **Shape:** `(batch_size, in_channels, num_spins, l_max, m_max)`
    *   `batch_size`: Number of samples in the batch.
    *   `in_channels`: Number of input feature maps.
    *   `num_spins`: The number of spin components in the feature representation.
    *   `l_max`, `m_max`: Dimensions representing the coefficients in the spectral domain (spherical harmonic degrees and orders).
*   **Data Type:** Complex-valued tensor (e.g., `torch.cfloat` or `tf.complex64`).
*   **Domain:** The input tensor must be in the **spectral domain**.

### 3. Output Tensor Specification

The module must produce an output tensor with the following characteristics:

*   **Shape:** `(batch_size, out_channels, num_spins, l_max_out, m_max_out)`
    *   `out_channels`: Number of output feature maps.
    *   `l_max_out`, `m_max_out`: The dimensions of the output in the spectral domain. If spectral pooling is applied, these will be smaller than the input `l_max` and `m_max`.
*   **Data Type:** Complex-valued tensor.
*   **Domain:** The output tensor is in the **spectral domain**.

### 4. Module Parameters and Constraints

A compliant module must be configurable with the following parameters:

*   `in_channels` (int): Number of input channels.
*   `out_channels` (int): Number of output channels.
*   `l_max_in` (int): The maximum spherical harmonic degree of the input.
*   `l_max_out` (int): The maximum spherical harmonic degree of the output. If `l_max_out < l_max_in`, spectral pooling is performed by truncating higher-degree coefficients.
*   `use_residual` (bool, optional): If `True`, a residual connection is added.
    *   **Constraint:** A residual connection is only possible if `in_channels == out_channels` and `l_max_in == l_max_out`. If these conditions are not met, the module must raise a configuration error or disable the residual connection.

### 5. Core Operations and Constraints

The internal implementation of the module must adhere to the following sequence of operations:

1.  **Spin-Weighted Spherical Convolution:**
    *   Must be performed in the spectral domain.
    *   A learnable filter with shape `(in_channels, out_channels, num_spins, l_max, m_max)` is required.
    *   The operation is an element-wise product between the input tensor and the learnable filter.

2.  **Spectral Batch Normalization:**
    *   Must be applied after the convolution.
    *   It must normalize the coefficients for each channel and spin across the batch.
    *   The `l=0` (mean) coefficient must be handled separately, typically by setting it to zero.

3.  **Phase Collapse Nonlinearity:**
    *   Must be applied after batch normalization.
    *   It must only modify the zero-spin (`s=0`) component of the feature maps.
    *   The non-zero spin components must pass through this stage unmodified.

4.  **Residual Connection (Optional):**
    *   If `use_residual` is `True`, the input to the block must be added to the output of the non-linearity.
    *   **Constraint:** This addition must occur in the **spectral domain**.

5.  **Spectral Pooling:**
    *   This is achieved by truncating the spherical harmonic coefficients. If `l_max_out` is less than `l_max_in`, the module must discard all coefficients with degree `l > l_max_out`.

### 6. Example Usage (Pseudocode)

```python
# Configuration
config = {
    "in_channels": 64,
    "out_channels": 128,
    "l_max_in": 32,
    "l_max_out": 16, # This implies spectral pooling
    "use_residual": False # Cannot use residual with pooling or channel change
}

# Initialize the block
spherical_conv_block = SphericalConvBlock(**config)

# Create a dummy input tensor (in spectral domain)
# Shape: (batch, 64, num_spins, 32, 32)
input_spectral = ...

# Forward pass
# Output shape: (batch, 128, num_spins, 16, 16)
output_spectral = spherical_conv_block(input_spectral)
```