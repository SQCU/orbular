"""
ssdf_encoder.py

This file contains the core logic for encoding a string of text into a
Spherical Signed Distance Field (S-SDF) and a binary mask.
It orchestrates the font, path, and geometry modules.
"""
import numpy as np
import ssdf_font
import ssdf_geometry as geom

def generate_stroke_geometries(text, path_func, type_size_rad=0.2):
    """
    Maps a string of text along a spherical path and generates the 3D stroke geometries.

    Args:
        text (str): The string to render.
        path_func (callable): A function `path(t)` that returns a 3D Cartesian point.
        type_size_rad (float): The angular height of the characters in radians.

    Returns:
        list: A list of tuples, where each tuple is (start_cart, end_cart) for a stroke.
    """
    all_strokes_cart = []
    num_letters = len(text)
    if num_letters == 0:
        return []

    for i, char in enumerate(text):
        if char.upper() not in ssdf_font.VECTOR_FONT:
            continue

        # Determine the letter's center point and tangent on the path
        t_center = (i + 0.5) / num_letters
        letter_center_cart = path_func(t_center)
        
        # Get local coordinate system (tangent plane)
        u_vec, v_vec = geom.get_orthonormal_vectors(letter_center_cart)

        # Map the 2D font strokes from the definition onto the tangent plane
        for p1_2d, p2_2d in ssdf_font.VECTOR_FONT[char.upper()]:
            # Scale by type size and project onto sphere
            stroke_start_cart = letter_center_cart + (p1_2d[0] * u_vec + p1_2d[1] * v_vec) * (type_size_rad / 2)
            stroke_end_cart = letter_center_cart + (p2_2d[0] * u_vec + p2_2d[1] * v_vec) * (type_size_rad / 2)
            
            # Normalize back to the unit sphere surface
            stroke_start_cart /= np.linalg.norm(stroke_start_cart)
            stroke_end_cart /= np.linalg.norm(stroke_end_cart)
            
            all_strokes_cart.append((stroke_start_cart, stroke_end_cart))
            
    return all_strokes_cart

def encode_to_sdf_and_mask(stroke_geometries, grid_size=128, stroke_thickness_rad=0.02):
    """
    Generates the binary mask and the ground truth S-SDF from stroke geometries.

    Args:
        stroke_geometries (list): A list of 3D stroke tuples (start_cart, end_cart).
        grid_size (int): The resolution of the output spherical grid (grid_size x grid_size).
        stroke_thickness_rad (float): The angular thickness for the binary mask.

    Returns:
        tuple: (mask_grid, sdf_grid) as two `(grid_size, grid_size)` numpy arrays.
    """
    # Create the spherical grid for sampling
    theta = np.linspace(0, np.pi, grid_size)
    phi = np.linspace(0, 2 * np.pi, grid_size)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    # Convert the entire grid to Cartesian coordinates for efficient calculation
    grid_cart_flat = geom.spherical_to_cartesian(theta_grid.ravel(), phi_grid.ravel())
    
    # Calculate the SDF for each point on the grid
    sdf_flat = np.full(grid_cart_flat.shape[0], np.inf)
    
    if not stroke_geometries: # Handle empty text case
        sdf_grid = sdf_flat.reshape((grid_size, grid_size)).T
        mask_grid = np.zeros_like(sdf_grid)
        return mask_grid, sdf_grid

    for i, p_cart in enumerate(grid_cart_flat):
        min_dist = np.inf
        for arc_start, arc_end in stroke_geometries:
            dist = geom.dist_point_to_arc(p_cart, arc_start, arc_end)
            if dist < min_dist:
                min_dist = dist
        sdf_flat[i] = min_dist
        
    # Reshape the flat SDF array into the grid
    sdf_grid = sdf_flat.reshape((grid_size, grid_size)).T
    
    # Generate the binary mask by thresholding the SDF
    mask_grid = (sdf_grid < stroke_thickness_rad).astype(np.float32)
    
    return mask_grid, sdf_grid
