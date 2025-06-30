
"""
sphere_plotter.py

A standardized library for plotting spherical data, including S0 (scalar) and
S2 (spin-2) fields. This module provides a consistent set of visualization
rules to be used across all demo and evaluation scripts.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import ssdf_decoder

# --- Helper Functions ---

def _to_numpy(tensor):
    """Converts a torch tensor to a numpy array, detaching and moving to CPU."""
    return tensor.detach().cpu().numpy()

def _get_convex_hull_view(mask):
    """
    Calculates the camera position to view the center of a mask's convex hull.
    """
    if mask.sum() == 0:
        return 30, -45 # Default view if no mask

    grid_shape = mask.shape
    grid_size_theta, grid_size_phi = grid_shape

    theta = np.linspace(0, np.pi, grid_size_theta)
    phi = np.linspace(0, 2 * np.pi, grid_size_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)

    points = np.stack([x[mask > 0], y[mask > 0], z[mask > 0]], axis=1)
    if points.shape[0] < 4:
        return 30, -45 # Not enough points for a hull

    hull = ConvexHull(points)
    center = np.mean(hull.points[hull.vertices], axis=0)
    
    # Convert center to spherical coordinates for camera angles
    r = np.linalg.norm(center)
    elev = np.rad2deg(np.arccos(center[2] / r))
    azim = np.rad2deg(np.arctan2(center[1], center[0]))
    
    return elev, azim

# --- 2D Plotting Functions ---

def plot_s0_panel(ax, s0_tensor, title, cmap='gray_r'):
    """Plots a 2D equirectangular view of a scalar field (e.g., bitmask)."""
    ax.imshow(_to_numpy(s0_tensor), cmap=cmap)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

def plot_sdf_panel(ax, sdf_tensor, title, cmap='coolwarm'):
    """Plots a 2D equirectangular view of an S-SDF."""
    im = ax.imshow(_to_numpy(sdf_tensor), cmap=cmap)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def plot_s2_panel(ax, s2_tensor, title, cmap='viridis'):
    """Plots the magnitude of a 2D equirectangular spin-2 field."""
    s2_mag = torch.abs(s2_tensor)
    im = ax.imshow(_to_numpy(s2_mag), cmap=cmap)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# --- 3D Plotting Functions ---

def plot_3d_reconstruction(ax, s0_sdf, s2_strain=None, title="3D Reconstruction", elev=None, azim=None):
    """
    Renders the extruded text from an S-SDF on a 3D sphere, optionally
    deformed by an S2 strain field.
    """
    device = s0_sdf.device
    s0_sdf_np = _to_numpy(s0_sdf)
    decoded_bitmask_np = ssdf_decoder.decode_sdf_to_texture(s0_sdf_np)
    decoded_bitmask = torch.from_numpy(decoded_bitmask_np).to(device)

    if decoded_bitmask.sum() == 0:
        ax.text(0.5, 0.5, 0.5, "No Surface", ha='center', va='center')
        ax.set_axis_off()
        return

    if elev is None or azim is None:
        elev, azim = _get_convex_hull_view(decoded_bitmask_np)

    grid_shape = s0_sdf.shape
    grid_size_theta, grid_size_phi = grid_shape

    theta = torch.linspace(0, np.pi, grid_size_theta, device=device)
    phi = torch.linspace(0, 2 * np.pi, grid_size_phi, device=device)
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')

    x_sphere = torch.sin(theta_grid) * torch.cos(phi_grid)
    y_sphere = torch.sin(theta_grid) * torch.sin(phi_grid)
    z_sphere = torch.cos(theta_grid)

    extrusion_factor = 0.05
    extruded_surface = torch.stack([x_sphere, y_sphere, z_sphere], dim=-1)
    extrusion = extruded_surface * decoded_bitmask.unsqueeze(-1) * extrusion_factor
    
    final_surface = extruded_surface + extrusion

    if s2_strain is not None:
        u_vec = torch.stack([-torch.sin(phi_grid), torch.cos(phi_grid), torch.zeros_like(phi_grid)], dim=-1)
        v_vec = torch.stack(
            [torch.cos(theta_grid) * torch.cos(phi_grid),
             torch.cos(theta_grid) * torch.sin(phi_grid),
             -torch.sin(theta_grid)], dim=-1)
        strain_real = s2_strain.real.unsqueeze(-1)
        strain_imag = s2_strain.imag.unsqueeze(-1)
        deformation = (u_vec * strain_real + v_vec * strain_imag) * 0.1 * decoded_bitmask.unsqueeze(-1)
        final_surface += deformation

    points_to_plot = final_surface[decoded_bitmask > 0]
    points_to_plot_np = _to_numpy(points_to_plot)
    x, y, z = points_to_plot_np[:, 0], points_to_plot_np[:, 1], points_to_plot_np[:, 2]

    ax.scatter(x, y, z, c=z, cmap='viridis', s=1)
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)

from scipy.ndimage import label

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_3d_strained_sheath(ax, s0_sdf, s2_strain, title="Strained Sheath", elev=None, azim=None):
    """
    Renders the S0 object's point cloud and envelops it in a translucent
    "sheath" whose geometry is determined by the S2 strain field.
    """
    device = s0_sdf.device
    s0_sdf_np = _to_numpy(s0_sdf)
    decoded_bitmask_np = ssdf_decoder.decode_sdf_to_texture(s0_sdf_np)
    decoded_bitmask = torch.from_numpy(decoded_bitmask_np).to(device)

    if decoded_bitmask.sum() == 0:
        ax.text(0.5, 0.5, 0.5, "No Surface", ha='center', va='center')
        ax.set_axis_off()
        return

    if elev is None or azim is None:
        elev, azim = _get_convex_hull_view(decoded_bitmask_np)

    grid_shape = s0_sdf.shape
    grid_size_theta, grid_size_phi = grid_shape

    theta = torch.linspace(0, np.pi, grid_size_theta, device=device)
    phi = torch.linspace(0, 2 * np.pi, grid_size_phi, device=device)
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')

    x_sphere = torch.sin(theta_grid) * torch.cos(phi_grid)
    y_sphere = torch.sin(theta_grid) * torch.sin(phi_grid)
    z_sphere = torch.cos(theta_grid)
    
    base_surface = torch.stack([x_sphere, y_sphere, z_sphere], dim=-1)

    # --- Render the base S0 object point cloud ---
    points_to_plot = base_surface[decoded_bitmask > 0]
    points_to_plot_np = _to_numpy(points_to_plot)
    ax.scatter(points_to_plot_np[:, 0], points_to_plot_np[:, 1], points_to_plot_np[:, 2], c='black', s=2, alpha=0.6)

    # --- Calculate deformation for the sheath ---
    u_vec = torch.stack([-torch.sin(phi_grid), torch.cos(phi_grid), torch.zeros_like(phi_grid)], dim=-1)
    v_vec = torch.stack(
        [torch.cos(theta_grid) * torch.cos(phi_grid),
         torch.cos(theta_grid) * torch.sin(phi_grid),
         -torch.sin(theta_grid)], dim=-1)
    strain_real = s2_strain.real.unsqueeze(-1)
    strain_imag = s2_strain.imag.unsqueeze(-1)
    deformation_vec = (u_vec * strain_real + v_vec * strain_imag) * 0.2 # Scale factor for visualization

    # --- Isolate and process each connected component ---
    labeled_mask, num_features = label(decoded_bitmask_np)
    for i in range(1, num_features + 1):
        component_mask = torch.from_numpy(labeled_mask == i).to(device)
        
        component_points = base_surface[component_mask]
        component_deformation = deformation_vec[component_mask]

        p_positive = component_points + component_deformation
        p_negative = component_points - component_deformation
        combined_points = torch.cat([p_positive, p_negative], dim=0)
        
        hull_points = _to_numpy(combined_points)
        if hull_points.shape[0] < 4: continue
        
        try:
            hull = ConvexHull(hull_points)
            # Create a collection of polygons for the hull faces
            faces = Poly3DCollection([hull_points[simplex] for simplex in hull.simplices])
            faces.set_facecolor('cyan')
            faces.set_edgecolor('k')
            faces.set_alpha(0.2)
            ax.add_collection3d(faces)
        except Exception as e:
            print(f"Could not compute hull for component {i}: {e}")

    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)



# --- Figure-Level Functions ---

def create_standard_figure(s0_texture, s0_sdf, s2_field, fig_title="Standard Visualization"):
    """
    Creates a standard 2x3 visualization figure:
    - Top Row: S0 Texture, S0 S-SDF, S2 Field Magnitude (2D)
    - Bottom Row: 3D Reconstruction, 3D Strained Sheath
    """
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(fig_title, fontsize=16)
    
    # 2D Plots
    ax1 = fig.add_subplot(2, 3, 1)
    plot_s0_panel(ax1, s0_texture, "S0 Texture (Input)")

    ax2 = fig.add_subplot(2, 3, 2)
    plot_sdf_panel(ax2, s0_sdf, "S0 S-SDF")

    ax3 = fig.add_subplot(2, 3, 3)
    plot_s2_panel(ax3, s2_field, "S2 Field (Magnitude)")

    # 3D Plots
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    plot_3d_reconstruction(ax4, s0_sdf, s2_field, "3D Deformed Reconstruction")
    
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    plot_3d_strained_sheath(ax5, s0_sdf, s2_field, "3D Strained Sheath")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig
