# gemini-2.5 first pass code, under review
"""
loss_visualization_demo_optimized.py

This script is the optimized version of the loss visualization demo.
It uses the new `ssdf_encoder_optimized` module to generate the S-SDFs,
which should be significantly faster due to vectorized calculations.
The rest of the logic for comparison and visualization remains the same.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import time

# Import the new modular and OPTIMIZED components
import ssdf_encoder_optimized as encoder
import ssdf_decoder as decoder
import ssdf_paths as paths
import ssdf_geometry as geom
import ssdf_loss as loss

def plot_sphere(ax, texture, title, cmap='gray_r', norm=None):
    """Helper function to plot a texture on a 3D sphere."""
    L = texture.shape[0]
    theta = np.linspace(0, np.pi, L)
    phi = np.linspace(0, 2 * np.pi, 2 * L - 1)
    theta, phi = np.meshgrid(theta, phi)

    # We need to remap the texture to the grid
    # The texture is (L, 2L-1), which matches the (theta, phi) grid
    # but phi needs to be wrapped for plotting.
    texture_remapped = np.zeros((2 * L, L))
    texture_remapped[:2*L-1, :] = texture.T

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Use the provided colormap and normalization
    facecolors = plt.get_cmap(cmap)(norm(texture_remapped)) if norm else plt.get_cmap(cmap)(texture_remapped)

    ax.plot_surface(x, y, z, facecolors=facecolors, rstride=1, cstride=1, antialiased=False, shade=False)
    ax.set_title(title)
    ax.set_axis_off()
    ax.view_init(elev=30, azim=-45)


def generate_sdf_for_path(text, grid_shape, path_start_sph, path_end_sph, type_size_rad, stroke_thickness_rad):
    """Helper function to generate an SDF for a given path using the OPTIMIZED encoder."""
    start_cart = geom.spherical_to_cartesian(*path_start_sph)
    end_cart = geom.spherical_to_cartesian(*path_end_sph)
    path_func = paths.great_circle_arc(start_cart, end_cart)
    
    stroke_geometries = encoder.generate_stroke_geometries(text, path_func, type_size_rad)
    
    # Use the optimized encoder function
    _, sdf = encoder.encode_to_sdf_and_mask_optimized(stroke_geometries, grid_shape, stroke_thickness_rad)
    
    return sdf

def main():
    """Main function to generate and visualize the loss between two SDFs."""
    
    # --- 1. Define Generation Parameters ---
    params = {
        "text": "SHIFT",
        "grid_size": 256,
        "path_start_sph": (np.pi / 2.5, np.pi / 4),
        "path_end_sph": (np.pi / 2.5, 7 * np.pi / 4),
        "type_size_rad": 0.3,
        "stroke_thickness_rad": 0.025,
        "translation_degrees": 5.0
    }
    
    print(f"Generating comparison for text: '{params['text']}' translated by {params['translation_degrees']} degrees.")

    # --- 2. Generate Reference S-SDF (Target) ---
    print("Generating reference S-SDF (optimized)...")
    start_time = time.time()
    target_sdf = generate_sdf_for_path(
        params["text"],
        (params["grid_size"], 2 * params["grid_size"] -1),
        params["path_start_sph"],
        params["path_end_sph"],
        params["type_size_rad"],
        params["stroke_thickness_rad"]
    )
    print(f"  -> Reference SDF generated in {time.time() - start_time:.2f} seconds.")

    # --- 3. Generate Translated S-SDF (Predicted) ---
    print("Generating translated S-SDF (optimized)...")
    start_time = time.time()
    translation_rad = np.deg2rad(params["translation_degrees"])
    
    translated_start_sph = (params["path_start_sph"][0], params["path_start_sph"][1] + translation_rad)
    translated_end_sph = (params["path_end_sph"][0], params["path_end_sph"][1] + translation_rad)
    
    predicted_sdf = generate_sdf_for_path(
        params["text"],
        (params["grid_size"], 2 * params["grid_size"] -1),
        translated_start_sph,
        translated_end_sph,
        params["type_size_rad"],
        params["stroke_thickness_rad"]
    )
    print(f"  -> Translated SDF generated in {time.time() - start_time:.2f} seconds.")

    # --- 4. Calculate Loss ---
    scalar_loss, error_surface = loss.mean_squared_error_sdf(predicted_sdf, target_sdf)
    print(f"MSE between reference and translated SDF: {scalar_loss:.6f}")

    # --- 5. Decode for Visualization ---
    target_texture = decoder.decode_sdf_to_texture(target_sdf, threshold=params["stroke_thickness_rad"])
    predicted_texture = decoder.decode_sdf_to_texture(predicted_sdf, threshold=params["stroke_thickness_rad"])

    # --- 6. Plotting ---
    print("Plotting results...")
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(3, 2)
    fig.suptitle(f"Optimized S-SDF Loss Visualization ({params['translation_degrees']}° Shift)", fontsize=18)

    # 2D plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    ax1.imshow(target_texture, cmap='gray_r', extent=[0, 360, 0, 180])
    ax1.set_title("Target (Reference Texture)")
    ax1.set_xlabel("Longitude (deg)")
    ax1.set_ylabel("Latitude (deg)")

    ax2.imshow(predicted_texture, cmap='gray_r', extent=[0, 360, 0, 180])
    ax2.set_title("Predicted (Translated Texture)")
    ax2.set_xlabel("Longitude (deg)")

    overlay = np.stack([target_texture, predicted_texture, np.zeros_like(target_texture)], axis=-1)
    ax3.imshow(overlay, extent=[0, 360, 0, 180])
    ax3.set_title("Overlay (Red=Target, Green=Predicted)")
    ax3.set_xlabel("Longitude (deg)")
    ax3.set_ylabel("Latitude (deg)")

    im = ax4.imshow(error_surface, cmap='inferno', extent=[0, 360, 0, 180])
    ax4.set_title(f"Error Surface (MSE: {scalar_loss:.4f})")
    ax4.set_xlabel("Longitude (deg)")
    fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    # 3D plots
    ax5 = fig.add_subplot(gs[2, 0], projection='3d')
    plot_sphere(ax5, predicted_texture, "Prediction as Spherical Texture")

    ax6 = fig.add_subplot(gs[2, 1], projection='3d')
    norm = Normalize(vmin=error_surface.min(), vmax=error_surface.max())
    plot_sphere(ax6, error_surface, "Loss Surface as Spherical Texture", cmap='inferno', norm=norm)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_filename = "ssdf_loss_visualization_optimized.png"
    plt.savefig(output_filename)
    
    print(f"Optimized loss visualization saved to '{output_filename}'")
    plt.show()

if __name__ == '__main__':
    main()
