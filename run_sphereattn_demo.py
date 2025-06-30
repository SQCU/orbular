# gemini-2.5 first pass code, under review
"""
run_sphereattn_demo.py

This script demonstrates the SphericalMultiHeadAttention layer by generating
a target S-SDF, using the attention layer to make a prediction, and
visualizing the results.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

import ssdf_encoder_optimized as encoder
import ssdf_decoder as decoder
import ssdf_paths as paths
import ssdf_geometry as geom
import ssdf_loss as loss
from spherical_attention import SphericalMultiHeadAttention
import s2fft

def generate_sdf_for_path(text, grid_shape, path_start_sph, path_end_sph, type_size_rad, stroke_thickness_rad):
    """Helper function to generate an SDF for a given path."""
    start_cart = geom.spherical_to_cartesian(*path_start_sph)
    end_cart = geom.spherical_to_cartesian(*path_end_sph)
    path_func = paths.great_circle_arc(start_cart, end_cart)
    
    stroke_geometries = encoder.generate_stroke_geometries(text, path_func, type_size_rad)
    
    _, sdf = encoder.encode_to_sdf_and_mask_optimized(stroke_geometries, grid_shape, stroke_thickness_rad)
    
    return sdf

def plot_sphere(ax, texture, title, cmap='gray_r', norm=None):
    """Helper function to plot a texture on a 3D sphere."""
    L = texture.shape[0]
    theta = np.linspace(0, np.pi, L)
    phi = np.linspace(0, 2 * np.pi, 2 * L - 1)
    theta, phi = np.meshgrid(theta, phi)

    texture_remapped = np.zeros((2 * L, L))
    texture_remapped[:2*L-1, :] = texture.T

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    facecolors = plt.get_cmap(cmap)(norm(texture_remapped)) if norm else plt.get_cmap(cmap)(texture_remapped)

    ax.plot_surface(x, y, z, facecolors=facecolors, rstride=1, cstride=1, antialiased=False, shade=False)
    ax.set_title(title)
    ax.set_axis_off()
    ax.view_init(elev=30, azim=-45)

def main():
    # --- 1. Define Generation Parameters ---
    params = {
        "text": "ATTN",
        "grid_size": 64,
        "path_start_sph": (np.pi / 3, np.pi / 4),
        "path_end_sph": (np.pi / 3, 7 * np.pi / 4),
        "type_size_rad": 0.4,
        "stroke_thickness_rad": 0.03,
    }
    
    # --- 2. Model Parameters ---
    L = params["grid_size"]
    num_heads = 4
    in_channels = 4
    out_channels = 1

    # --- 3. Generate Target S-SDF ---
    print("Generating target S-SDF...")
    target_sdf = generate_sdf_for_path(
        params["text"],
        (L, 2 * L - 1),
        params["path_start_sph"],
        params["path_end_sph"],
        params["type_size_rad"],
        params["stroke_thickness_rad"]
    )
    target_sdf_lm = torch.from_numpy(s2fft.forward(target_sdf, L=L)).unsqueeze(0).unsqueeze(0)

    # --- 4. Create Model and Random Input ---
    model = SphericalMultiHeadAttention(in_channels, out_channels, L, num_heads)
    random_input_lm = torch.randn(1, in_channels, L, 2 * L -1, dtype=torch.complex64)

    # --- 5. Get Prediction ---
    print("Running forward pass...")
    predicted_sdf_lm = model(random_input_lm)
    predicted_sdf = s2fft.inverse(predicted_sdf_lm.squeeze(0).squeeze(0).detach().numpy(), L=L).real

    # --- 6. Calculate Loss ---
    scalar_loss, error_surface = loss.mean_squared_error_sdf(predicted_sdf, target_sdf)
    scalar_loss = scalar_loss.real
    error_surface = error_surface.real
    print(f"MSE Loss: {scalar_loss}")

    # --- 7. Decode for Visualization ---
    target_texture = decoder.decode_sdf_to_texture(target_sdf, threshold=params["stroke_thickness_rad"])
    predicted_texture = decoder.decode_sdf_to_texture(predicted_sdf, threshold=params["stroke_thickness_rad"])

    # --- 8. Plotting ---
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3)
    fig.suptitle("Spherical Multi-Head Attention Demo", fontsize=18)

    # SDF plots
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(target_sdf, cmap='viridis')
    ax1.set_title("Target SDF")
    fig.colorbar(im1, ax=ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(predicted_sdf, cmap='viridis')
    ax2.set_title("Predicted SDF")
    fig.colorbar(im2, ax=ax2)

    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(error_surface, cmap='inferno')
    ax3.set_title(f"Loss Surface (MSE: {scalar_loss:.4f})")
    fig.colorbar(im3, ax=ax3)

    # Sphere plots
    ax4 = fig.add_subplot(gs[1, 0], projection='3d')
    plot_sphere(ax4, target_texture, "Target Texture")

    ax5 = fig.add_subplot(gs[1, 1], projection='3d')
    plot_sphere(ax5, predicted_texture, "Predicted Texture")

    ax6 = fig.add_subplot(gs[1, 2], projection='3d')
    norm = Normalize(vmin=error_surface.min(), vmax=error_surface.max())
    plot_sphere(ax6, error_surface, "Loss as Spherical Texture", cmap='inferno', norm=norm)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("sphereattn_demo_output.png")
    plt.show()


if __name__ == '__main__':
    main()
