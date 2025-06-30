# gemini-2.5 first pass code, under review
"""
ssdf_generator.py

This script serves as the main entry point for generating and visualizing
Spherical Signed Distance Field (S-SDF) data. It demonstrates how to use the
modular components (font, paths, geometry, encoder, decoder, loss) to create
a complete data sample and visualize the results, including the error surface.

This script is a counterpart to the original `s_we_generator.py`, but it is
built upon the new, modular, and extensible S-SDF framework.
"""
import numpy as np
import matplotlib.pyplot as plt

# Import the new modular components
import ssdf_encoder as encoder
import ssdf_decoder as decoder
import ssdf_paths as paths
import ssdf_geometry as geom
import ssdf_loss as loss

def main():
    """Main function to generate and visualize S-SDF data."""
    
    # --- 1. Define Generation Parameters (The "Compact Input") ---
    generation_params = {
        "text": "SSDF",
        "grid_size": 256,
        "path_type": "great_circle",
        "path_start_sph": (np.pi / 3, np.pi / 4),
        "path_end_sph": (2 * np.pi / 3, 5 * np.pi / 3),
        "type_size_rad": 0.4,
        "stroke_thickness_rad": 0.03
    }
    
    print(f"Generating S-SDF for text: '{generation_params['text']}'...")

    # --- 2. Generate Path Function ---
    start_cart = geom.spherical_to_cartesian(*generation_params["path_start_sph"])
    end_cart = geom.spherical_to_cartesian(*generation_params["path_end_sph"])
    path_func = paths.great_circle_arc(start_cart, end_cart)

    # --- 3. Generate Stroke Geometries ---
    stroke_geometries = encoder.generate_stroke_geometries(
        generation_params["text"],
        path_func,
        generation_params["type_size_rad"]
    )

    # --- 4. Encode to S-SDF and Mask (Ground Truth) ---
    input_mask, target_sdf = encoder.encode_to_sdf_and_mask(
        stroke_geometries,
        generation_params["grid_size"],
        generation_params["stroke_thickness_rad"]
    )

    # --- 5. Simulate a Model Prediction (e.g., by adding noise) ---
    # In a real scenario, this would come from a trained model.
    # Here, we simulate it to demonstrate the loss visualization.
    noise = np.random.normal(0, 0.01, target_sdf.shape)
    predicted_sdf = target_sdf + noise

    # --- 6. Calculate Loss ---
    scalar_loss, error_surface = loss.mean_squared_error_sdf(predicted_sdf, target_sdf)
    print(f"Simulated MSE Loss: {scalar_loss:.6f}")

    # --- 7. Decode for Visualization ---
    decoded_texture_from_target = decoder.decode_sdf_to_texture(target_sdf, threshold=generation_params["stroke_thickness_rad"])
    decoded_texture_from_pred = decoder.decode_sdf_to_texture(predicted_sdf, threshold=generation_params["stroke_thickness_rad"])

    # --- 8. Plotting ---
    print("Generation and simulation complete. Plotting results...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Modular S-SDF Generation: '{generation_params['text']}'", fontsize=18)

    # Plot Input Mask
    ax1 = axes[0, 0]
    ax1.imshow(input_mask, cmap='gray_r', extent=[0, 360, 0, 180])
    ax1.set_title("Input: Binary Mask")
    ax1.set_xlabel("Longitude (deg)")
    ax1.set_ylabel("Latitude (deg)")

    # Plot Target SDF
    ax2 = axes[0, 1]
    im2 = ax2.imshow(target_sdf, cmap='viridis', extent=[0, 360, 0, 180])
    ax2.set_title("Target: Ground Truth S-SDF")
    ax2.set_xlabel("Longitude (deg)")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Plot Decoded Prediction
    ax3 = axes[1, 0]
    ax3.imshow(decoded_texture_from_pred, cmap='gray_r', extent=[0, 360, 0, 180])
    ax3.set_title("Decoded Texture from Prediction")
    ax3.set_xlabel("Longitude (deg)")
    ax3.set_ylabel("Latitude (deg)")

    # Plot Error Surface
    ax4 = axes[1, 1]
    im4 = ax4.imshow(error_surface, cmap='inferno', extent=[0, 360, 0, 180])
    ax4.set_title(f"Error Surface (MSE: {scalar_loss:.4f})")
    ax4.set_xlabel("Longitude (deg)")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_filename = "ssdf_modular_visualization.png"
    plt.savefig(output_filename)
    
    print(f"Visualization saved to '{output_filename}'")
    plt.show()


if __name__ == '__main__':
    main()
