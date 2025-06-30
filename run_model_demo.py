# gemini-2.5 first pass code, under review
"""
run_model_demo.py

This script demonstrates the full pipeline:
1. Generates a sample (input mask and target S-SDF) using the optimized encoder.
2. Instantiates the `SimpleSphericalCNN` with random initial weights.
3. Performs a forward pass of the input mask through the model.
4. Calculates the loss between the model''s output and the ground truth S-SDF.
5. Visualizes the input, the raw model output, the ground truth, and the
   resulting error surface, providing a complete picture of the untrained
   model's performance.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import time

# Import our project modules
import ssdf_encoder_optimized as encoder
import ssdf_decoder as decoder
import ssdf_paths as paths
import ssdf_geometry as geom
import ssdf_loss as loss
from spherical_cnn import SimpleSphericalCNN

def main():
    """Main function to run the model demo."""
    
    # --- 1. Define Generation and Model Parameters ---
    # The bandlimit L determines the grid size for MW sampling, which is (2L, 2L-1)
    # We'll use a bandlimit that results in a grid close to our original target.
    L = 32 
    GRID_SIZE_THETA = 2 * L
    GRID_SIZE_PHI = 2 * L -1
    TEXT = "CNN"
    
    print(f"--- S-SDF Model Demo ---")
    print(f"Bandlimit L: {L}, Grid Size: {GRID_SIZE_THETA}x{GRID_SIZE_PHI}, Text: '{TEXT}'")

    # --- 2. Generate the Data ---
    print("\nStep 1: Generating data...")
    start_time = time.time()
    
    path_start_sph = (np.pi / 3, np.pi / 4)
    path_end_sph = (2 * np.pi / 3, 5 * np.pi / 3)
    type_size_rad = 0.5
    stroke_thickness_rad = 0.05

    start_cart = geom.spherical_to_cartesian(*path_start_sph)
    end_cart = geom.spherical_to_cartesian(*path_end_sph)
    path_func = paths.great_circle_arc(start_cart, end_cart)
    
    stroke_geometries = encoder.generate_stroke_geometries(TEXT, path_func, type_size_rad)
    
    # Note: We are generating on a grid that is not the native MW grid,
    # but s2fft will handle the resampling. This is not ideal for performance
    # but it's the most direct way to test the pipeline.
    input_mask, target_sdf = encoder.encode_to_sdf_and_mask_optimized(
        stroke_geometries,
        (GRID_SIZE_THETA, GRID_SIZE_PHI),
        stroke_thickness_rad
    )
    print(f"  -> Data generated in {time.time() - start_time:.2f} seconds.")

    # --- 3. Prepare Tensors for PyTorch ---
    # Convert numpy arrays to PyTorch tensors and add batch & channel dimensions
    input_tensor = torch.from_numpy(input_mask).unsqueeze(0).unsqueeze(0).float()
    target_tensor = torch.from_numpy(target_sdf).unsqueeze(0).unsqueeze(0).float()

    # --- 4. Instantiate and Run the Model ---
    print("\nStep 2: Initializing and running the Spherical CNN...")
    start_time = time.time()
    
    # Instantiate the model with random weights
    model = SimpleSphericalCNN(L=L)
    
    # Set model to evaluation mode (disables things like dropout if they were present)
    model.eval()
    
    # Perform the forward pass
    with torch.no_grad(): # We don't need to track gradients for this demo
        predicted_sdf_tensor = model(input_tensor)
        
    print(f"  -> Model forward pass completed in {time.time() - start_time:.2f} seconds.")

    # --- 5. Upsample and Calculate Loss ---
    print("\nStep 3: Upsampling model output and calculating loss...")
    
    # Upsample the model's output to match the target shape
    upsample_size = (target_tensor.shape[2], target_tensor.shape[3])
    predicted_sdf_upsampled = torch.nn.functional.interpolate(
        predicted_sdf_tensor, 
        size=upsample_size, 
        mode='bilinear', 
        align_corners=False
    )

    # Convert predicted tensor back to numpy for our loss function
    predicted_sdf_np = predicted_sdf_upsampled.squeeze(0).squeeze(0).numpy()
    
    # --- DEBUG: Print shapes before loss calculation ---
    print(f"  -> Shape of Predicted SDF (after upsampling): {predicted_sdf_np.shape}")
    print(f"  -> Shape of Target SDF:                       {target_sdf.shape}")
    
    scalar_loss, error_surface = loss.mean_squared_error_sdf(predicted_sdf_np, target_sdf)
    print(f"  -> MSE Loss (Random Model vs. Ground Truth): {scalar_loss:.6f}")

    # --- 6. Plotting ---
    print("\nStep 4: Plotting results...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Untrained Spherical CNN Output vs. Ground Truth", fontsize=18)

    # Plot Input Mask
    axes[0, 0].imshow(input_mask, cmap='gray_r', extent=[0, 360, 0, 180])
    axes[0, 0].set_title("Input to CNN (Binary Mask)")
    axes[0, 0].set_xlabel("Longitude (deg)")
    axes[0, 0].set_ylabel("Latitude (deg)")

    # Plot Ground Truth SDF
    im2 = axes[0, 1].imshow(target_sdf, cmap='viridis', extent=[0, 360, 0, 180])
    axes[0, 1].set_title("Target (Ground Truth S-SDF)")
    axes[0, 1].set_xlabel("Longitude (deg)")
    fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Plot Model's Raw Output
    im3 = axes[1, 0].imshow(predicted_sdf_np, cmap='viridis', extent=[0, 360, 0, 180])
    axes[1, 0].set_title("Model Output (from Random Weights)")
    axes[1, 0].set_xlabel("Longitude (deg)")
    axes[1, 0].set_ylabel("Latitude (deg)")
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Plot Error Surface
    im4 = axes[1, 1].imshow(error_surface, cmap='inferno', extent=[0, 360, 0, 180])
    axes[1, 1].set_title(f"Error Surface (MSE: {scalar_loss:.4f})")
    axes[1, 1].set_xlabel("Longitude (deg)")
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_filename = "spherical_cnn_demo_output.png"
    plt.savefig(output_filename)
    
    print(f"\nDemo complete. Visualization saved to '{output_filename}'")
    plt.show()

if __name__ == '__main__':
    main()
