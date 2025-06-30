# gemini-2.5 first pass code, under review
"""
jelly_letter_demo.py

This script generates and visualizes data for the (s0, s2) "Jelly Letter" problem.
The ground truth s2 field is synthesized from a physically-motivated model based
on inter-letter repulsion, making it a more meaningful target for a model to learn.
"""
import numpy as np
import s2fft
from s2fft.utils.rotation import rotate_flms
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

# Import necessary components from the project
import ssdf_encoder_optimized as encoder
import ssdf_decoder as decoder
import ssdf_paths as paths
import ssdf_geometry as geom

# --- 1. Data Generation ---

def generate_jelly_letter_data(text, L, sampling="mw"):
    """
    Generates a single data sample for the (s0, s2) Jelly Letter problem.

    This creates:
    1. A spin-0 field representing an n-gram shape (from its SDF).
    2. A physically-motivated spin-2 stress field derived from the shape's
       repulsion potential.

    Args:
        text (str): The text to render (e.g., "SPIN").
        L (int): The spherical harmonic bandlimit.
        sampling (str): The s2fft sampling scheme.

    Returns:
        A tuple containing the s0 spatial field, s2 spatial field, s0 spectral
        field, and s2 spectral field.
    """
    # --- Generate base s0 shape and its Signed Distance Field (SDF) ---
    path_start_sph = (np.pi / 2.5, np.pi / 4)
    path_end_sph = (np.pi / 2.5, 7 * np.pi / 4)
    start_cart = geom.spherical_to_cartesian(*path_start_sph)
    end_cart = geom.spherical_to_cartesian(*path_end_sph)
    path_func = paths.great_circle_arc(start_cart, end_cart)

    stroke_geometries = encoder.generate_stroke_geometries(text, path_func, type_size_rad=0.3)
    grid_shape = (L, 2 * L - 1)
    s0_spatial, s0_sdf = encoder.encode_to_sdf_and_mask_optimized(stroke_geometries, grid_shape, stroke_thickness_rad=0.025)
    s0_lm = s2fft.forward(s0_spatial, L=L, sampling=sampling)

    # --- Synthesize the Target s2 Stress Field from the s0 shape ---
    # 1. Create a scalar "repulsion potential" from the SDF.
    #    The potential is high near the letters.
    repulsion_potential = np.exp(-s0_sdf * 20.0)
    repulsion_lm = s2fft.forward(repulsion_potential, L=L, sampling=sampling)

    # 2. Define a spin-2 kernel that translates potential into stress.
    #    This kernel is defined across a range of l-modes for robustness,
    #    ensuring it overlaps with the signal from the repulsion potential.
    s2_kernel_lm = np.zeros_like(s0_lm, dtype=np.complex128)
    l_values = np.arange(L)
    # A simple filter that is stronger at low frequencies to create smooth stress
    l_filter = np.exp(-l_values / (L / 8.0))

    # We apply this filter to the m=2 and m=-2 modes, which are characteristic
    # of spin-2 fields, ensuring the correct conjugate symmetry for a real field.
    m_index_pos = L - 1 + 2
    m_index_neg = L - 1 - 2

    for l in range(2, L):  # For spin s, we must have l >= s
        # This ensures kernel_l,2 = conj(kernel_l,-2)
        s2_kernel_lm[l, m_index_pos] = l_filter[l] * (1.0 + 1.5j)
        s2_kernel_lm[l, m_index_neg] = l_filter[l] * (1.0 - 1.5j)

    # 3. Convolve the potential with the kernel to get the final s2 field.
    s2_lm = repulsion_lm * s2_kernel_lm
    s2_spatial = s2fft.inverse(s2_lm, L=L, sampling=sampling, spin=2)

    return s0_spatial, s2_spatial, s0_lm, s2_lm


# --- 2. Loss Calculation and Visualization ---

def visualize_jelly_letter_loss(target, prediction, loss, L, filename="jelly_letter_demo_output.png"):
    """
    Visualizes the target, prediction, and loss for the Jelly Letter problem.
    """
    fig, axs = plt.subplots(3, 3, figsize=(18, 15), dpi=120)
    fig.suptitle("Jelly Letter (s0, s2) Problem: Target, Prediction, and Loss", fontsize=20)

    # --- Plotting Titles ---
    axs[0, 0].set_title("Target: s0 Texture (Spatial)", fontsize=14)
    axs[0, 1].set_title("Target: s2 Stress Re(LM)", fontsize=14)
    axs[0, 2].set_title("Target: s2 Stress Im(LM)", fontsize=14)

    axs[1, 0].set_title("Prediction: s0 Texture (Spatial)", fontsize=14)
    axs[1, 1].set_title("Prediction: s2 Stress Re(LM)", fontsize=14)
    axs[1, 2].set_title("Prediction: s2 Stress Im(LM)", fontsize=14)

    axs[2, 0].set_title("Loss: s0 Shape (Spatial)", fontsize=14)
    axs[2, 1].set_title("Loss: s2 Stress (Spectral)", fontsize=14)
    axs[2, 2].set_title("Loss: s2 Stress (Spatial)", fontsize=14)

    # --- Decode s0 fields for cleaner visualization ---
    target_s0_tex = decoder.decode_sdf_to_texture(target['s0_spatial'], threshold=0.5)
    pred_s0_tex = decoder.decode_sdf_to_texture(prediction['s0_spatial'], threshold=0.5)

    # --- Row 1: Target ---
    im = axs[0, 0].imshow(target_s0_tex, cmap='gray_r'); fig.colorbar(im, ax=axs[0, 0])
    im = axs[0, 1].imshow(np.real(target['s2_lm']), cmap='coolwarm', norm=SymLogNorm(linthresh=1e-3)); fig.colorbar(im, ax=axs[0, 1])
    im = axs[0, 2].imshow(np.imag(target['s2_lm']), cmap='coolwarm', norm=SymLogNorm(linthresh=1e-3)); fig.colorbar(im, ax=axs[0, 2])

    # --- Row 2: Prediction ---
    im = axs[1, 0].imshow(pred_s0_tex, cmap='gray_r'); fig.colorbar(im, ax=axs[1, 0])
    im = axs[1, 1].imshow(np.real(prediction['s2_lm']), cmap='coolwarm', norm=SymLogNorm(linthresh=1e-3)); fig.colorbar(im, ax=axs[1, 1])
    im = axs[1, 2].imshow(np.imag(prediction['s2_lm']), cmap='coolwarm', norm=SymLogNorm(linthresh=1e-3)); fig.colorbar(im, ax=axs[1, 2])

    # --- Row 3: Loss ---
    im = axs[2, 0].imshow(loss['s0_loss_spatial'], cmap='viridis'); fig.colorbar(im, ax=axs[2, 0])
    im = axs[2, 1].imshow(loss['s2_loss_lm'], cmap='viridis', norm=SymLogNorm(linthresh=1e-5)); fig.colorbar(im, ax=axs[2, 1])
    s2_loss_spatial = s2fft.inverse(loss['s2_loss_lm'], L=L, sampling="mw")
    im = axs[2, 2].imshow(np.real(s2_loss_spatial), cmap='viridis'); fig.colorbar(im, ax=axs[2, 2])

    for ax in axs.flat:
        ax.set_xlabel("m")
        ax.set_ylabel("l")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename)
    print(f"Saved visualization to {filename}")


# --- 3. Main Execution ---

if __name__ == '__main__':
    # --- Parameters ---
    L = 128
    SAMPLING = "mw"
    TEXT = "SPIN"
    MAX_ROT_DEG = 2.0

    # --- Generate Target Data ---
    print(f"Generating target data for text: '{TEXT}'...")
    s0_target_spatial, _, s0_target_lm, s2_target_lm = generate_jelly_letter_data(TEXT, L, SAMPLING)

    target_data = {'s0_spatial': s0_target_spatial, 's2_lm': s2_target_lm}

    # --- Simulate a Prediction with a Small Randomized Rotation ---
    print(f"Simulating a prediction with a small random rotation (max {MAX_ROT_DEG}°)...")
    alpha = np.deg2rad(np.random.uniform(-MAX_ROT_DEG, MAX_ROT_DEG))
    beta = np.deg2rad(np.random.uniform(-MAX_ROT_DEG, MAX_ROT_DEG))
    gamma = np.deg2rad(np.random.uniform(-MAX_ROT_DEG, MAX_ROT_DEG))

    # Apply spectral rotation to the s2 field
    s2_pred_lm = rotate_flms(s2_target_lm, L, (alpha, beta, gamma))

    # Apply a corresponding spatial shift to the s0 field
    phi_shift = int(L * (alpha + gamma) / np.pi)
    theta_shift = int(L * beta / np.pi)
    s0_pred_spatial = np.roll(s0_target_spatial, shift=(theta_shift, phi_shift), axis=(0, 1))

    prediction_data = {'s0_spatial': s0_pred_spatial, 's2_lm': s2_pred_lm}

    # --- Calculate Loss ---
    print("Calculating loss fields...")
    s0_loss_spatial = (s0_pred_spatial - s0_target_spatial)**2
    s2_loss_lm = np.abs(s2_pred_lm - s2_target_lm)**2

    loss_data = {'s0_loss_spatial': s0_loss_spatial, 's2_loss_lm': s2_loss_lm}

    # --- Visualize ---
    print("Generating visualization...")
    visualize_jelly_letter_loss(target_data, prediction_data, loss_data, L)