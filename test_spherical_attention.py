# gemini-2.5 first pass code, under review
"""
test_spherical_attention.py

This script contains a simple test for the SphericalMultiHeadAttention layer.
It checks if the layer can be initialized and if the forward pass
produces an output of the correct shape.
"""
import torch
from spherical_attention import SphericalMultiHeadAttention

def test_spherical_mha():
    """
    Tests the SphericalMultiHeadAttention layer's initialization and forward pass.
    """
    # --- 1. Define Test Parameters ---
    batch_size = 2
    L = 16  # Bandlimit
    num_heads = 4
    in_channels = 32  # Must be divisible by num_heads
    out_channels = 64
    
    print("--- Testing SphericalMultiHeadAttention ---")
    print(f"Parameters: L={L}, Heads={num_heads}, In Channels={in_channels}, Out Channels={out_channels}")

    # --- 2. Initialize the Layer ---
    try:
        model = SphericalMultiHeadAttention(
            in_channels=in_channels,
            out_channels=out_channels,
            L=L,
            num_heads=num_heads
        )
        print("✅ Layer initialized successfully.")
    except Exception as e:
        print(f"❌ Layer initialization failed: {e}")
        return

    # --- 3. Create a Random Input Tensor ---
    # The input is in the spectral domain (complex coefficients)
    input_shape = (batch_size, in_channels, L, 2 * L - 1)
    f_in_lm = torch.randn(*input_shape, dtype=torch.complex64)
    print(f"Input tensor created with shape: {f_in_lm.shape}")

    # --- 4. Perform a Forward Pass ---
    try:
        output_lm = model(f_in_lm)
        print("✅ Forward pass completed.")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return

    # --- 5. Check Output Shape ---
    expected_shape = (batch_size, out_channels, L, 2 * L - 1)
    print(f"Expected output shape: {expected_shape}")
    print(f"Actual output shape:   {output_lm.shape}")

    assert output_lm.shape == expected_shape, \
        f"Shape mismatch! Expected {expected_shape}, got {output_lm.shape}"
    
    print("✅ Output shape is correct.")
    print("--- Test Passed ---")

if __name__ == '__main__':
    test_spherical_mha()
