# gemini-2.5 first pass code, under review
"""
loss_visualization_demo.py

This script provides a synthetic but structured comparison to visualize the
output of the S-SDF loss function.

It works by:
1. Generating a reference S-SDF for a given text along a specific path.
2. Generating a second "predicted" S-SDF where the text has been translated
   by a few degrees across the sphere's surface.
3. Computing the Mean Squared Error between these two SDFs.
4. Visualizing the resulting error surface, which should clearly highlight
   the leading and trailing edges of the translation, providing a palpable
   sense of the error function.
"""
import numpy as np
import matplotlib.pyplot as plt

import ssdf_encoder as encoder
import ssdf_decoder as decoder
import ssdf_paths as paths
import ssdf_geometry as geom
import ssdf_loss as loss

def generate_sdf_for_path(text, grid_size, path_start_sph, path_end_sph, type_size_rad, stroke_thickness_rad):
    """Helper function to generate an SDF for a given path."""
    start_cart = geom.spherical_to_cartesian(*path_start_sph)
    end_cart = geom.spherical_to_cartesian(*path_end_sph)
    path_func = paths.great_circle_arc(start_cart, end_cart)
    
    stroke_geometries = encoder.generate_stroke_geometries(text, path_func, type_size_rad)
    
    _, sdf = encoder.encode_to_sdf_and_mask(stroke_geometries, grid_size, stroke_thickness_rad)
    
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
    print("Generating reference S-SDF...")
    target_sdf = generate_sdf_for_path(
        params["text"],
        params["grid_size"],
        params["path_start_sph"],
        params["path_end_sph"],
        params["type_size_rad"],
        params["stroke_thickness_rad"]
    )

    # --- 3. Generate Translated S-SDF (Predicted) ---
    print("Generating translated S-SDF...")
    translation_rad = np.deg2rad(params["translation_degrees"])
    
    # Translate by shifting the longitude of the path points
    translated_start_sph = (params["path_start_sph"][0], params["path_start_sph"][1] + translation_rad)
    translated_end_sph = (params["path_end_sph"][0], params["path_end_sph"][1] + translation_rad)
    
    predicted_sdf = generate_sdf_for_path(
        params["text"],
        params["grid_size"],
        translated_start_sph,
        translated_end_sph,
        params["type_size_rad"],
        params["stroke_thickness_rad"]
    )

    # --- 4. Calculate Loss ---
    scalar_loss, error_surface = loss.mean_squared_error_sdf(predicted_sdf, target_sdf)
    print(f"MSE between reference and translated SDF: {scalar_loss:.6f}")

    # --- 5. Decode for Visualization ---
    target_texture = decoder.decode_sdf_to_texture(target_sdf, threshold=params["stroke_thickness_rad"])
    predicted_texture = decoder.decode_sdf_to_texture(predicted_sdf, threshold=params["stroke_thickness_rad"])

    # --- 6. Plotting ---
    print("Plotting results...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"S-SDF Loss Visualization via Translation ({params['translation_degrees']}° Shift)", fontsize=18)

    # Plot Target Texture
    axes[0, 0].imshow(target_texture, cmap='gray_r', extent=[0, 360, 0, 180])
    axes[0, 0].set_title("Target (Reference Texture)")
    axes[0, 0].set_xlabel("Longitude (deg)")
    axes[0, 0].set_ylabel("Latitude (deg)")

    # Plot Predicted Texture
    axes[0, 1].imshow(predicted_texture, cmap='gray_r', extent=[0, 360, 0, 180])
    axes[0, 1].set_title("Predicted (Translated Texture)")
    axes[0, 1].set_xlabel("Longitude (deg)")

    # Plot Overlay
    # Use different colors for the overlay
    overlay = np.stack([target_texture, predicted_texture, np.zeros_like(target_texture)], axis=-1)
    axes[1, 0].imshow(overlay, extent=[0, 360, 0, 180])
    axes[1, 0].set_title("Overlay (Red=Target, Green=Predicted)")
    axes[1, 0].set_xlabel("Longitude (deg)")
    axes[1, 0].set_ylabel("Latitude (deg)")

    # Plot Error Surface
    im = axes[1, 1].imshow(error_surface, cmap='inferno', extent=[0, 360, 0, 180])
    axes[1, 1].set_title(f"Error Surface (MSE: {scalar_loss:.4f})")
    axes[1, 1].set_xlabel("Longitude (deg)")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_filename = "ssdf_loss_visualization.png"
    plt.savefig(output_filename)
    
    print(f"Loss visualization saved to '{output_filename}'")
    plt.show()

if __name__ == '__main__':
    main()
