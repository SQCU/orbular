
"""
test_sphere_plotter.py

A test script to demonstrate the functionality of the `sphere_plotter` library.
This script generates random S0 and S2 fields and uses the standardized
plotting functions to create a visualization.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import sphere_plotter

def generate_random_data(grid_shape=(128, 256)):
    """Generates random s0 and s2 fields for testing."""
    # Create a random s0 texture (a few circles)
    s0_texture = torch.zeros(grid_shape)
    for _ in range(3):
        c_theta = np.random.randint(20, grid_shape[0] - 20)
        c_phi = np.random.randint(20, grid_shape[1] - 20)
        radius = np.random.randint(10, 20)
        
        theta = torch.arange(grid_shape[0]).unsqueeze(1)
        phi = torch.arange(grid_shape[1])
        
        dist_sq = (theta - c_theta)**2 + (phi - c_phi)**2
        s0_texture[dist_sq < radius**2] = 1.0

    # Create a simple SDF from the texture (very basic)
    from scipy.ndimage import distance_transform_edt
    s0_sdf_np = distance_transform_edt(1 - _to_numpy(s0_texture)) - distance_transform_edt(_to_numpy(s0_texture))
    s0_sdf = torch.from_numpy(s0_sdf_np)

    # Create a random s2 field
    s2_field = torch.randn(grid_shape, dtype=torch.complex64)

    return s0_texture, s0_sdf, s2_field

def _to_numpy(tensor):
    """Helper to convert tensor to numpy."""
    return tensor.detach().cpu().numpy()

if __name__ == '__main__':
    print("Generating random data for visualization test...")
    s0_texture, s0_sdf, s2_field = generate_random_data()

    print("Creating standard visualization figure...")
    fig = sphere_plotter.create_standard_figure(
        s0_texture,
        s0_sdf,
        s2_field,
        fig_title="Sphere Plotter Test"
    )

    output_filename = "sphere_plotter_test_output.png"
    plt.savefig(output_filename)
    print(f"Saved test visualization to {output_filename}")
    plt.show()
