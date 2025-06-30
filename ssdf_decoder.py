"""
ssdf_decoder.py

Contains the function to decode a Spherical Signed Distance Field (S-SDF)
back into a visual, binary texture. This is primarily used for evaluation
and visualization of a model's output.
"""
import numpy as np

def decode_sdf_to_texture(sdf_tensor, threshold=0.02):
    """
    Converts an S-SDF tensor back to a binary texture for visualization.
    This function is the inverse of the SDF concept, turning distance back into shape.

    Args:
        sdf_tensor (np.ndarray): A 2D numpy array representing the S-SDF.
        threshold (float): A small value that defines the thickness of the
                           reconstructed strokes. Points with an SDF value
                           less than this are considered "inside" the shape.

    Returns:
        np.ndarray: A binary texture of the same shape as the input.
    """
    if not isinstance(sdf_tensor, np.ndarray):
        raise TypeError("Input sdf_tensor must be a numpy array.")
        
    return (sdf_tensor < threshold).astype(np.float32)
