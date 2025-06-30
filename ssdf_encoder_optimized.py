"""
ssdf_encoder_optimized.py

This file contains a faster, vectorized version of the S-SDF encoder.
It leverages NumPy's array operations to avoid slow, explicit Python loops,
making the S-SDF generation process significantly more performant.
A progress bar using tqdm is also integrated for user feedback.
"""
import numpy as np
import tqdm
import ssdf_font
import ssdf_geometry as geom

def generate_stroke_geometries(text, path_func, type_size_rad=0.2):
    """
    Maps a string of text along a spherical path and generates the 3D stroke geometries.
    (This function is identical to the one in the original encoder but is included
    here for completeness of the module.)
    """
    all_strokes_cart = []
    num_letters = len(text)
    if num_letters == 0:
        return []

    for i, char in enumerate(text):
        if char.upper() not in ssdf_font.VECTOR_FONT:
            continue

        t_center = (i + 0.5) / num_letters
        letter_center_cart = path_func(t_center)
        u_vec, v_vec = geom.get_orthonormal_vectors(letter_center_cart)

        for p1_2d, p2_2d in ssdf_font.VECTOR_FONT[char.upper()]:
            stroke_start_cart = letter_center_cart + (p1_2d[0] * u_vec + p1_2d[1] * v_vec) * (type_size_rad / 2)
            stroke_end_cart = letter_center_cart + (p2_2d[0] * u_vec + p2_2d[1] * v_vec) * (type_size_rad / 2)
            
            stroke_start_cart /= np.linalg.norm(stroke_start_cart)
            stroke_end_cart /= np.linalg.norm(stroke_end_cart)
            
            all_strokes_cart.append((stroke_start_cart, stroke_end_cart))
            
    return all_strokes_cart

def dist_grid_to_arc_vectorized(grid_cart_flat, arc_start_cart, arc_end_cart):
    """
    Calculates the shortest great-circle distance from every point in a grid
    to a single great-circle arc using vectorized NumPy operations.
    """
    # Normal to the plane defined by the arc
    arc_plane_normal = np.cross(arc_start_cart, arc_end_cart)
    norm = np.linalg.norm(arc_plane_normal)
    if np.isclose(norm, 0):
        # If arc is a point, calculate distance to that point
        return np.arccos(np.clip(np.dot(grid_cart_flat, arc_start_cart), -1.0, 1.0))

    arc_plane_normal /= norm

    # Dot product of all grid points with the arc endpoints
    dot_start = np.dot(grid_cart_flat, arc_start_cart)
    dot_end = np.dot(grid_cart_flat, arc_end_cart)

    # Check if the closest point on the great circle lies within the arc segment
    # This is true if the grid point is "between" the planes defined by the endpoints
    # and perpendicular to the arc plane. A simpler check is based on angles.
    arc_length = np.arccos(np.clip(np.dot(arc_start_cart, arc_end_cart), -1.0, 1.0))
    dist_to_start = np.arccos(np.clip(dot_start, -1.0, 1.0))
    dist_to_end = np.arccos(np.clip(dot_end, -1.0, 1.0))

    # Condition to check if the projection is on the arc
    on_arc_condition = (dist_to_start <= arc_length) & (dist_to_end <= arc_length)

    # Distance to the great circle itself
    dist_to_gc = np.abs(np.pi / 2 - np.arccos(np.clip(np.dot(grid_cart_flat, arc_plane_normal), -1.0, 1.0)))
    
    # If not on the arc, the closest point is one of the endpoints
    dist_to_endpoints = np.minimum(dist_to_start, dist_to_end)

    return np.where(on_arc_condition, dist_to_gc, dist_to_endpoints)


def encode_to_sdf_and_mask_optimized(stroke_geometries, grid_size=128, stroke_thickness_rad=0.02):
    """
    Generates the binary mask and the ground truth S-SDF from stroke geometries
    using an optimized, vectorized approach.
    """
    theta = np.linspace(0, np.pi, grid_size)
    phi = np.linspace(0, 2 * np.pi, grid_size)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    grid_cart_flat = geom.spherical_to_cartesian(theta_grid.ravel(), phi_grid.ravel())
    
    # Initialize SDF to infinity
    sdf_flat = np.full(grid_cart_flat.shape[0], np.inf)
    
    if not stroke_geometries:
        sdf_grid = sdf_flat.reshape((grid_size, grid_size)).T
        mask_grid = np.zeros_like(sdf_grid)
        return mask_grid, sdf_grid

    # Iterate through each stroke, calculating the distance for all grid points at once
    for arc_start, arc_end in tqdm.tqdm(stroke_geometries, desc="Encoding Strokes", unit="stroke"):
        dist_to_arc = dist_grid_to_arc_vectorized(grid_cart_flat, arc_start, arc_end)
        # Update the SDF with the minimum distance found so far
        sdf_flat = np.minimum(sdf_flat, dist_to_arc)
        
    sdf_grid = sdf_flat.reshape((grid_size, grid_size)).T
    mask_grid = (sdf_grid < stroke_thickness_rad).astype(np.float32)
    
    return mask_grid, sdf_grid
