
# Code Review: Spherical CNN vs. Spherical Multi-Head Attention

## Introduction

This code review analyzes the implementations of a Spherical Convolutional Neural Network (`spherical_cnn.py`) and a Spherical Multi-Head Attention mechanism (`spherical_attention.py`). The review is based on the provided source code, the paper "Scaling Spherical CNNs" (Esteves et al., 2023), and the visual outputs from the demo scripts.

The core idea behind this review is to move beyond line-by-line code correctness and instead focus on how to instrument these models to produce verifiable logs. These logs should allow us to ask and answer specific questions about the models' behavior, particularly concerning anomalies and performance.

## High-Level Comparison

| Feature | Spherical CNN (`spherical_cnn.py`) | Spherical Multi-Head Attention (`spherical_attention.py`) |
|---|---|---|
| **Core Operation** | Convolution in the spherical harmonic domain. | Self-attention in the spherical harmonic domain. |
| **Complexity** | `O(N * C_in * C_out * L^2)` per layer, where N is batch size, C is channels, L is bandlimit. | `O(N * H * C_h * L^2)` where H is number of heads and C_h is head dimension. |
| **Receptive Field**| Local, defined by the kernel size in the spatial domain. | Global, as every point can attend to every other point. |
| **Strengths** | Efficient for learning local patterns, translationally equivariant. | Good at capturing long-range dependencies and complex relationships. |
| **Weaknesses** | May struggle with long-range dependencies. | Computationally more expensive, especially at high resolutions. |

## Analysis of `spherical_cnn.py`

The `SimpleSphericalCNN` implementation is a straightforward application of the principles described in the paper. It uses `s2fft` to perform convolutions in the harmonic domain, which is the correct and efficient way to do it.

### Suggestions for Logging and Analysis

1.  **Kernel Visualization**: The convolutional kernels are learned parameters. It would be insightful to visualize them in the spatial domain to understand what features the network is learning. This can be done by inverse-transforming the learned `weight` tensor.

    *   **Logging Specification**: After each training epoch, or at regular intervals, perform an inverse `s2fft` on the `weight` tensor of each `SphericalConv2d` layer and log the resulting spatial kernel as an image. This will show the evolution of the learned filters.

2.  **Residual Flow Analysis**: The paper mentions the use of residual blocks. While the provided `SimpleSphericalCNN` doesn't use them, it's a crucial concept. For a deeper network, monitoring the flow of information through residual connections is vital.

    *   **Logging Specification**: In a version of the network with residual blocks, log the norm of the residual and the norm of the main path's output before they are added together. A histogram of the ratio of these norms over a batch would reveal if the network is primarily relying on the skip connections (vanishing gradients in the main path) or if the convolutional blocks are learning meaningful transformations.

3.  **Activation Statistics**: The choice of activation function (ReLU in this case) is critical.

    *   **Logging Specification**: Log a histogram of the activations after each `SphericalConv2d` layer. This can help identify "dead neurons" (neurons that always output zero) and saturation issues.

## Analysis of `spherical_attention.py`

The `SphericalMultiHeadAttention` module is a more complex and novel piece of architecture. It adapts the multi-head attention mechanism, popularized by Transformers, to the spherical domain. The implementation correctly performs the attention mechanism in the harmonic domain.

### Suggestions for Logging and Analysis

1.  **Attention Pattern Visualization**: The core of the attention mechanism is the attention pattern, `A_h_spatial`. Visualizing this is key to understanding the model's behavior.

    *   **Logging Specification**: For a given input, log the `A_h_spatial` for each head as a heatmap on the sphere. This will show which parts of the input the model is "paying attention to" when producing the output. Averaging these attention maps over a validation set can reveal systematic patterns.

2.  **Entropy of Attention**: A sharp, low-entropy attention map indicates that the model is focusing on a small, specific region. A diffuse, high-entropy map suggests a more global aggregation of information.

    *   **Logging Specification**: Calculate the entropy of each `A_h_spatial` distribution. Logging the average entropy per head over time can show if the model learns to specialize its heads, with some focusing on local details (low entropy) and others on global context (high entropy).

3.  **Q, K, V Analysis**: The queries (Q), keys (K), and values (V) are the building blocks of attention.

    *   **Logging Specification**: Log the norm of the `F_Q_h`, `F_K_h`, and `F_V_h` tensors. This can help diagnose if any of these components are collapsing to zero, which would indicate a problem with the learning process.

## Self-Review of Visual Outputs

**Disclaimer**: This portion of the review is a self-assessment of the generated code's output without comparison to a canonical, known-correct implementation. The goal is to deduce the correctness of the SO(3) operations from first principles and visual inspection of the output images.

### Analysis of `spherical_cnn_demo_output.png` and `sphereattn_demo_output.png`

The key question is whether the visual outputs from two different random initializations of the networks suggest a correct implementation of the spectral domain SO(3) operations.

1.  **Isotropy and Smoothness**: With random weights, a spherical convolution should act as an isotropic (rotationally invariant) filter. This means the output should be smooth and not biased towards any particular direction on the sphere. Both `spherical_cnn_demo_output.png` and `sphereattn_demo_output.png` exhibit this property. The patterns are complex but do not show any obvious grid artifacts or preferred directions that would suggest an error in the spherical harmonic transforms.

2.  **Consistency with Theory**:
    *   **Spherical CNN**: The output of the CNN is, as expected, a set of smooth, localized features. The random kernels have been applied across the sphere, resulting in a pattern that is statistically similar everywhere. This is consistent with the properties of spherical convolutions.
    *   **Spherical Attention**: The attention mechanism, even with random weights, should produce a more complex, non-local output. The `sphereattn_demo_output.png` shows this. The attention mechanism is relating different parts of the sphere, leading to a more intricate pattern than the simple convolution. The fact that this intricate pattern is still smooth and isotropic is a strong positive signal.

3.  **Lack of High-Frequency Noise**: A common failure mode in spectral methods is the appearance of high-frequency noise or ringing artifacts. The outputs in both images are smooth and do not show these artifacts. This suggests that the `s2fft` forward and inverse transforms are being used correctly, and the data is being handled properly in the spectral domain.

### Conclusion of Self-Review

Based on the visual evidence from the two output images, the implementations of both the Spherical CNN and the Spherical Multi-Head Attention appear to be correctly implementing the spectral domain SO(3) operations. The outputs are consistent with the theoretical properties of these operations when applied with random weights.

To be more certain, one could perform a more rigorous test:

*   **Rotation Equivariance Test**: Apply a known rotation to the input signal and check if the output of the network rotates by the same amount. This would be a definitive test of the SO(3) equivariance. This could be implemented as a unit test, where the rotated output is compared to the original output with a small tolerance.
