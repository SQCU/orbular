
"""
s0_s2_pipeline.py

This script implements an end-to-end pipeline for generating, comparing,
and visualizing spin-0 (S-SDF) and spin-2 (strain) fields on the sphere.

The pipeline is as follows:
1. Synthesize a bitmask and an arbitrary autorepulsion field (spin-2).
2. Synthesize a signed distance field (spin-0) from the bitmask.
3. Synthesize a perturbed field by applying a rotation to the initial data.
4. Calculate and plot the loss between the original and perturbed fields for both s0 and s2.
5. Plot a 3D reconstruction of the original data, showing the letters
   extruded from the sphere and deformed by the s2 strain field. It also
   visualizes the loss field as a "tangled" reconstruction.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import existing modules
import ssdf_paths
import ssdf_geometry as geom
import ssdf_encoder_optimized as ssdf_encoder
import ssdf_loss
import ssdf_decoder

# Use torch for tensor operations from the start
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Core Synthesis Functions ---

def get_rotation_matrix(axis, angle):
    """Returns the 3D rotation matrix for a given axis and angle."""
    axis = torch.tensor(axis, dtype=torch.float32, device=device)
    axis = axis / torch.linalg.norm(axis)
    angle = torch.tensor(angle, dtype=torch.float32, device=device)
    a = torch.cos(angle / 2.0)
    b, c, d = -axis * torch.sin(angle / 2.0)
    return torch.tensor([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
        [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]
    ], device=device)

def synthesize_fields(text, path_func, grid_shape, type_size_rad=0.2, stroke_thickness_rad=0.02, sanity_magnitude=1):
    """
    Synthesizes s0 (SDF) and s2 (strain) fields from text embossed on a sphere.
    This function is designed to be feed-forward and uses torch tensors.
    """
    # 1. Generate stroke geometries (using the numpy-based encoder for now)
    stroke_geometries = ssdf_encoder.generate_stroke_geometries(
        text, path_func, type_size_rad=type_size_rad, sanity_magnitude=sanity_magnitude
    )

    # 2. Encode to S-SDF and bitmask
    bitmask_np, s0_sdf_np = ssdf_encoder.encode_to_sdf_and_mask_optimized(
        stroke_geometries, grid_shape, stroke_thickness_rad=stroke_thickness_rad
    )
    bitmask = torch.from_numpy(bitmask_np).to(device)
    s0_sdf = torch.from_numpy(s0_sdf_np).to(device)

    # 3. Synthesize the spin-2 "autorepulsion" field
    grid_size_theta, grid_size_phi = grid_shape
    theta = torch.linspace(0, np.pi, grid_size_theta, device=device)
    phi = torch.linspace(0, 2 * np.pi, grid_size_phi, device=device)
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')

    # Strain magnitude is strongest inside the letters and decays away
    strain_magnitude = torch.exp(-s0_sdf * 5.0) * bitmask

    # Strain orientation (phase) is based on the angle of the gradient of the SDF
    s0_grad_theta, s0_grad_phi = torch.gradient(s0_sdf)
    strain_phase = torch.atan2(s0_grad_phi, s0_grad_theta)
    
    s2_strain_field = strain_magnitude * torch.exp(1j * 2 * strain_phase)

    # --- Sanity Checks ---
    if sanity_magnitude > 0:
        print(f"Synthesized Fields for '{text}':")
        print(f"  S0 SDF min: {s0_sdf.min():.4f}, max: {s0_sdf.max():.4f}")
        print(f"  S2 Strain mag min: {torch.abs(s2_strain_field).min():.4f}, max: {torch.abs(s2_strain_field).max():.4f}")
        print(f"  Bitmask sum: {bitmask.sum()}")

    return bitmask, s0_sdf, s2_strain_field

def create_perturbed_fields(text, path_func, grid_shape, rotation_axis, rotation_angle_rad, **kwargs):
    """
    Creates a second set of fields by rotating the original path.
    """
    rot_matrix = get_rotation_matrix(rotation_axis, rotation_angle_rad)

    def perturbed_path(t):
        # The original path_func returns numpy arrays, so we convert for matrix multiplication
        original_point = torch.from_numpy(path_func(t)).to(torch.float32).to(device)
        rotated_point = torch.matmul(rot_matrix, original_point)
        return rotated_point.cpu().numpy()

    return synthesize_fields(text, perturbed_path, grid_shape, **kwargs)


# --- Visualization Functions ---

def plot_loss_fields(s0_loss, s2_loss, s0_loss_field, s2_loss_field):
    """Plots the scalar loss values and the 2D loss fields."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Loss Visualization", fontsize=16)

    im0 = axs[0].imshow(s0_loss_field.cpu().numpy(), cmap='magma')
    axs[0].set_title(f"S0 (SDF) Loss Field\nTotal Loss: {s0_loss:.6f}")
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(torch.abs(s2_loss_field).cpu().numpy(), cmap='magma')
    axs[1].set_title(f"S2 (Strain) Loss Field (Magnitude)\nTotal Loss: {np.abs(s2_loss):.6f}")
    fig.colorbar(im1, ax=axs[1])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("s0_s2_loss_visualization.png")
    plt.show()

def plot_reconstruction(s0_sdf, s2_strain, title, filename):
    """
    Creates a 3D visualization of the engraved letters on a sphere,
    deformed by the strain field, based on the decoded S-SDF.
    This function renders only the extruded text against a transparent background.
    """
    # Decode the S-SDF to get a clean bitmask
    decoded_bitmask_np = ssdf_decoder.decode_sdf_to_texture(s0_sdf.cpu().numpy())
    decoded_bitmask = torch.from_numpy(decoded_bitmask_np).to(device)

    if decoded_bitmask.sum() == 0:
        print(f"Warning: No surface to reconstruct for '{title}'. Skipping plot.")
        return

    grid_shape = s0_sdf.shape
    grid_size_theta, grid_size_phi = grid_shape

    theta = torch.linspace(0, np.pi, grid_size_theta, device=device)
    phi = torch.linspace(0, 2 * np.pi, grid_size_phi, device=device)
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')

    # Create sphere mesh
    x_sphere = torch.sin(theta_grid) * torch.cos(phi_grid)
    y_sphere = torch.sin(theta_grid) * torch.sin(phi_grid)
    z_sphere = torch.cos(theta_grid)

    # Extrude points within the bitmask along the surface normal
    extrusion_factor = 0.05
    extruded_surface = torch.stack([x_sphere, y_sphere, z_sphere], dim=-1)
    extrusion = extruded_surface * decoded_bitmask.unsqueeze(-1) * extrusion_factor

    # Deform the extruded surface based on the strain field
    u_vec = torch.stack([-torch.sin(phi_grid), torch.cos(phi_grid), torch.zeros_like(phi_grid)], dim=-1)
    v_vec = torch.stack(
        [torch.cos(theta_grid) * torch.cos(phi_grid),
         torch.cos(theta_grid) * torch.sin(phi_grid),
         -torch.sin(theta_grid)], dim=-1)
    strain_real = s2_strain.real.unsqueeze(-1)
    strain_imag = s2_strain.imag.unsqueeze(-1)
    deformation = (u_vec * strain_real + v_vec * strain_imag) * 0.1 * decoded_bitmask.unsqueeze(-1)

    final_surface = extruded_surface + extrusion + deformation
    
    # Select only the points that are part of the extruded surface for plotting
    points_to_plot = final_surface[decoded_bitmask > 0]
    
    # Move to CPU for plotting
    points_to_plot_np = points_to_plot.cpu().numpy()
    x, y, z = points_to_plot_np[:, 0], points_to_plot_np[:, 1], points_to_plot_np[:, 2]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=z, cmap='viridis', s=1)
    ax.set_title(title)
    ax.set_box_aspect([1,1,1])
    ax.axis('off') # Make axes invisible
    plt.savefig(filename, transparent=True)
    plt.show()

def main():
    """Executes the full s0-s2 generation and visualization pipeline."""
    # --- 1. Setup ---
    grid_shape = (128, 256)
    text = "S2"
    sanity_magnitude = 0 # Set to 0 to disable checks for final run

    # Define multiple discrete paths
    paths = [
        ssdf_paths.great_circle_arc(geom.spherical_to_cartesian(np.pi / 4, np.pi / 4), geom.spherical_to_cartesian(np.pi / 4, 3 * np.pi / 4)),
        ssdf_paths.great_circle_arc(geom.spherical_to_cartesian(3 * np.pi / 4, np.pi / 4), geom.spherical_to_cartesian(3 * np.pi / 4, 3 * np.pi / 4)),
        ssdf_paths.great_circle_arc(geom.spherical_to_cartesian(np.pi / 4, 5 * np.pi / 4), geom.spherical_to_cartesian(np.pi / 4, 7 * np.pi / 4)),
        ssdf_paths.great_circle_arc(geom.spherical_to_cartesian(3 * np.pi / 4, 5 * np.pi / 4), geom.spherical_to_cartesian(3 * np.pi / 4, 7 * np.pi / 4)),
    ]

    # --- 2. Synthesize Original and Perturbed Fields ---
    s0_orig_acc = torch.full(grid_shape, np.inf, device=device)
    s2_orig_acc = torch.zeros(grid_shape, dtype=torch.complex64, device=device)
    s0_pert_acc = torch.full(grid_shape, np.inf, device=device)
    s2_pert_acc = torch.zeros(grid_shape, dtype=torch.complex64, device=device)

    for i, path_func in enumerate(paths):
        print(f"Processing path {i+1}/{len(paths)}...")
        _, s0_orig, s2_orig = synthesize_fields(
            text, path_func, grid_shape, sanity_magnitude=sanity_magnitude, type_size_rad=0.1
        )
        _, s0_pert, s2_pert = create_perturbed_fields(
            text, path_func, grid_shape,
            rotation_axis=[0, 0, 1], rotation_angle_rad=np.deg2rad(5),
            sanity_magnitude=sanity_magnitude, type_size_rad=0.1
        )
        s0_orig_acc = torch.min(s0_orig_acc, s0_orig)
        s2_orig_acc += s2_orig
        s0_pert_acc = torch.min(s0_pert_acc, s0_pert)
        s2_pert_acc += s2_pert

    # --- 3. Calculate and Visualize Loss ---
    print("Step 3: Calculating and visualizing loss fields...")
    s0_loss, s0_loss_field = ssdf_loss.mean_squared_error_sdf(
        s0_pert_acc.cpu().numpy(), s0_orig_acc.cpu().numpy()
    )
    s2_loss, s2_loss_field = ssdf_loss.mean_squared_error_sdf(
        s2_pert_acc.cpu().numpy(), s2_orig_acc.cpu().numpy()
    )
    plot_loss_fields(s0_loss, s2_loss, torch.from_numpy(s0_loss_field), torch.from_numpy(s2_loss_field))

    # --- 4. Visualize Reconstructions ---
    print("Step 4: Generating 3D reconstructions...")
    
    # Original reconstruction
    plot_reconstruction(
        s0_orig_acc, s2_orig_acc,
        title="Reconstruction of Original Embossing",
        filename="s0_s2_reconstruction_original.png"
    )

    # Loss field reconstruction ("tangled letters")
    loss_s0_field = torch.from_numpy(s0_loss_field).to(device)
    loss_s2_field = torch.from_numpy(s2_loss_field).to(device)
    
    plot_reconstruction(
        loss_s0_field, loss_s2_field,
        title="Reconstruction of Loss Field ('Tangled Letters')",
        filename="s0_s2_reconstruction_loss.png"
    )

if __name__ == "__main__":
    main()
