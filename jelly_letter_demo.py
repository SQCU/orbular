
# gemini-2.5 first pass code, under review
"""
jelly_letter_demo.py

This script generates and visualizes data for the (s0, s2) "Jelly Letter" problem.
It demonstrates how to represent a coupled scalar (spin-0) and tensor (spin-2)
field on the sphere, and how to compute and visualize a loss function that is
sensitive to errors in both magnitude and orientation (spin).
"""
import numpy as np
import s2fft
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

# --- 1. Data Generation ---

def generate_jelly_letter_data(L: int, sampling: str = "mw"):
    """
    Generates a single data sample for the (s0, s2) Jelly Letter problem.

    This creates a spin-0 field representing a letter 'C' and a coupled
    spin-2 field representing the internal stress/strain of the jelly.

    Args:
        L (int): The spherical harmonic bandlimit.
        sampling (str): The s2fft sampling scheme.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - s0_spatial (np.ndarray): The spatial representation of the spin-0 letter shape.
            - s2_spatial (np.ndarray): The spatial representation of the spin-2 stress field.
    """
    # Create a grid for the letter shape
    s0_spatial = np.zeros((L, 2 * L - 1), dtype=np.float32)

    # Define a 'C' shape in grid coordinates
    center_l, center_m = L // 2, (2 * L - 1) // 2
    radius = L // 4
    thickness = L // 16
    for l in range(L):
        for m in range(2 * L - 1):
            dist_sq = (l - center_l)**2 + (m - center_m)**2
            if (radius - thickness)**2 < dist_sq < (radius + thickness)**2:
                # Check if it's in the 'C' part (not the gap)
                if m > center_m - radius // 2:
                    s0_spatial[l, m] = 1.0

    # Transform the letter shape to the spectral domain to create a coupled stress field
    s0_lm = s2fft.forward(s0_spatial, L=L, sampling=sampling)

    # Create a simple, handcrafted spin-2 kernel in the spectral domain
    # This kernel will "pinch" the jelly along a certain orientation
    s2_kernel_lm = np.zeros_like(s0_lm, dtype=np.complex128)
    s2_kernel_lm[2, L-1+2] = 1.0 + 1.0j  # m=2
    s2_kernel_lm[2, L-1-2] = 1.0 - 1.0j  # m=-2

    # Create the spin-2 stress field by convolving the shape with the kernel
    # (multiplication in the spectral domain)
    s2_lm = s0_lm * s2_kernel_lm

    # For visualization, transform the s2 field back to spatial domain
    s2_spatial = s2fft.inverse(s2_lm, L=L, sampling=sampling, spin=2)

    return s0_spatial, s2_spatial, s0_lm, s2_lm


# --- 2. Loss Calculation and Visualization ---

def visualize_jelly_letter_loss(target, prediction, loss, L, filename="jelly_letter_demo_output.png"):
    """
    Visualizes the target, prediction, and loss for the Jelly Letter problem.

    Args:
        target (dict): Dictionary containing target fields ('s0_spatial', 's2_lm').
        prediction (dict): Dictionary containing predicted fields ('s0_spatial', 's2_lm').
        loss (dict): Dictionary containing loss fields ('s0_loss_spatial', 's2_loss_lm').
        L (int): The spherical harmonic bandlimit.
        filename (str): The output filename.
    """
    fig, axs = plt.subplots(3, 3, figsize=(18, 15), dpi=120)
    fig.suptitle("Jelly Letter (s0, s2) Problem: Target, Prediction, and Loss", fontsize=20)

    # --- Plotting Titles ---
    axs[0, 0].set_title("Target: s0 Shape (Spatial)", fontsize=14)
    axs[0, 1].set_title("Target: s2 Stress Re(LM)", fontsize=14)
    axs[0, 2].set_title("Target: s2 Stress Im(LM)", fontsize=14)

    axs[1, 0].set_title("Prediction: s0 Shape (Spatial)", fontsize=14)
    axs[1, 1].set_title("Prediction: s2 Stress Re(LM)", fontsize=14)
    axs[1, 2].set_title("Prediction: s2 Stress Im(LM)", fontsize=14)

    axs[2, 0].set_title("Loss: s0 Shape (Spatial)", fontsize=14)
    axs[2, 1].set_title("Loss: s2 Stress (Spectral)", fontsize=14)
    axs[2, 2].set_title("Loss: s2 Stress (Spatial)", fontsize=14)


    # --- Row 1: Target ---
    im00 = axs[0, 0].imshow(target['s0_spatial'], cmap='magma')
    im01 = axs[0, 1].imshow(np.real(target['s2_lm']), cmap='coolwarm', norm=SymLogNorm(linthresh=1e-3))
    im02 = axs[0, 2].imshow(np.imag(target['s2_lm']), cmap='coolwarm', norm=SymLogNorm(linthresh=1e-3))
    fig.colorbar(im00, ax=axs[0, 0], fraction=0.046, pad=0.04)
    fig.colorbar(im01, ax=axs[0, 1], fraction=0.046, pad=0.04)
    fig.colorbar(im02, ax=axs[0, 2], fraction=0.046, pad=0.04)

    # --- Row 2: Prediction ---
    im10 = axs[1, 0].imshow(prediction['s0_spatial'], cmap='magma')
    im11 = axs[1, 1].imshow(np.real(prediction['s2_lm']), cmap='coolwarm', norm=SymLogNorm(linthresh=1e-3))
    im12 = axs[1, 2].imshow(np.imag(prediction['s2_lm']), cmap='coolwarm', norm=SymLogNorm(linthresh=1e-3))
    fig.colorbar(im10, ax=axs[1, 0], fraction=0.046, pad=0.04)
    fig.colorbar(im11, ax=axs[1, 1], fraction=0.046, pad=0.04)
    fig.colorbar(im12, ax=axs[1, 2], fraction=0.046, pad=0.04)

    # --- Row 3: Loss ---
    im20 = axs[2, 0].imshow(loss['s0_loss_spatial'], cmap='viridis')
    im21 = axs[2, 1].imshow(loss['s2_loss_lm'], cmap='viridis', norm=SymLogNorm(linthresh=1e-5))
    
    # For a better visualization of the spatial loss, transform the spectral loss back
    s2_loss_spatial = s2fft.inverse(loss['s2_loss_lm'], L=L, sampling="mw")
    im22 = axs[2, 2].imshow(np.real(s2_loss_spatial), cmap='viridis')

    fig.colorbar(im20, ax=axs[2, 0], fraction=0.046, pad=0.04)
    fig.colorbar(im21, ax=axs[2, 1], fraction=0.046, pad=0.04)
    fig.colorbar(im22, ax=axs[2, 2], fraction=0.046, pad=0.04)

    for ax in axs.flat:
        ax.set_xlabel("m")
        ax.set_ylabel("l")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename)
    print(f"Saved visualization to {filename}")


# --- 3. Main Execution ---

if __name__ == '__main__':
    # --- Parameters ---
    L = 64  # Bandlimit
    SAMPLING = "mw"
    ANGULAR_ERROR_DEG = 5.0 # Degrees to rotate the prediction for demo purposes

    # --- Generate Target Data ---
    print("Generating target data...")
    s0_target_spatial, s2_target_spatial, s0_target_lm, s2_target_lm = generate_jelly_letter_data(L, SAMPLING)

    target_data = {
        's0_spatial': s0_target_spatial,
        's2_lm': s2_target_lm
    }

    # --- Simulate a Prediction with Angular Error ---
    print(f"Simulating a prediction with a {ANGULAR_ERROR_DEG}-degree angular error...")
    # A rotation around the z-axis by alpha corresponds to multiplying the
    # harmonic coefficients by exp(-i * m * alpha).
    alpha_rad = np.deg2rad(ANGULAR_ERROR_DEG)
    m_indices = np.arange(-(L - 1), L)
    rotation_phasor = np.exp(-1j * m_indices * alpha_rad)

    # Apply the rotation to the spin-2 target to create the prediction
    s2_pred_lm = s2_target_lm * rotation_phasor
    
    # For this demo, assume the s0 prediction is perfect
    s0_pred_spatial = s0_target_spatial

    prediction_data = {
        's0_spatial': s0_pred_spatial,
        's2_lm': s2_pred_lm
    }

    # --- Calculate Loss ---
    print("Calculating loss fields...")
    # s0 loss is simple MSE in the spatial domain
    s0_loss_spatial = (s0_pred_spatial - s0_target_spatial)**2

    # s2 loss is the squared magnitude of the difference in the spectral domain
    s2_loss_lm = np.abs(s2_pred_lm - s2_target_lm)**2

    loss_data = {
        's0_loss_spatial': s0_loss_spatial,
        's2_loss_lm': s2_loss_lm
    }

    # --- Visualize ---
    print("Generating visualization...")
    visualize_jelly_letter_loss(target_data, prediction_data, loss_data, L)
