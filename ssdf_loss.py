# gemini-2.5 first pass code, under review
"""
ssdf_loss.py

Contains loss functions for training S-SDF models.
The key feature here is that the loss function itself returns a plottable
S-SDF, allowing for a direct, palpable visualization of the error surface.
"""
import numpy as np

def mean_squared_error_sdf(predicted_sdf, target_sdf):
    """
    Calculates the Mean Squared Error between two S-SDFs.

    The function returns two values:
    1. The aggregate scalar loss value for backpropagation.
    2. A plottable "error SDF" which is the per-pixel squared difference.
       This allows for visualizing where the model's prediction is least accurate.

    Args:
        predicted_sdf (np.ndarray): The S-SDF output by the model.
        target_sdf (np.ndarray): The ground truth S-SDF.

    Returns:
        tuple: (scalar_loss, error_sdf)
            - scalar_loss (float): The mean of the squared errors.
            - error_sdf (np.ndarray): A 2D array of the same shape as the inputs,
                                      representing the spatial distribution of error.
    """
    if predicted_sdf.shape != target_sdf.shape:
        raise ValueError("Predicted and target SDFs must have the same shape.")

    # Calculate the per-pixel squared difference
    error_sdf = (predicted_sdf - target_sdf)**2
    
    # Calculate the aggregate scalar loss
    scalar_loss = np.mean(error_sdf)
    
    return scalar_loss, error_sdf

# --- Stub for a more advanced loss function ---
def perceptual_sdf_loss(predicted_sdf, target_sdf):
    """
    STUB: A more advanced, perceptual loss for S-SDFs.

    This loss could, for example, weigh errors near the zero-crossing (the
    actual surface) more heavily than errors far away. It could also take
    into account the gradient of the SDF.

    Args:
        predicted_sdf (np.ndarray): The S-SDF output by the model.
        target_sdf (np.ndarray): The ground truth S-SDF.

    Returns:
        tuple: (scalar_loss, error_sdf)
    """
    # To be implemented:
    # 1. Define a weighting function, e.g., w(d) = exp(-alpha * abs(d)),
    #    where d is the distance from the target_sdf.
    # 2. Calculate the weighted squared error: w(target_sdf) * (pred - target)**2
    # 3. (Optional) Add a term for the difference in gradients:
    #    || grad(predicted_sdf) - grad(target_sdf) ||^2
    print("STUB: perceptual_sdf_loss function is not yet implemented.")
    
    # Fallback to MSE
    return mean_squared_error_sdf(predicted_sdf, target_sdf)
