"""
ssdf_geometry.py

Core, pure functions for spherical geometry and coordinate transformations.
These are fundamental utilities used by other S-SDF modules.
"""
import numpy as np

# --- Coordinate System Conversions ---
def spherical_to_cartesian(theta, phi):
    """
    Converts spherical coordinates (colatitude, longitude) to 3D Cartesian.
    Assumes a unit sphere (radius = 1).
    
    Args:
        theta (float or np.ndarray): Colatitude (angle from z-axis) in radians.
        phi (float or np.ndarray): Longitude (angle from x-axis in xy-plane) in radians.

    Returns:
        np.ndarray: The corresponding 3D Cartesian coordinates [x, y, z].
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    # Stacking for single points or arrays of points
    if hasattr(theta, '__len__'):
        return np.stack([x, y, z], axis=-1)
    return np.array([x, y, z])

def cartesian_to_spherical(p_cart):
    """
    Converts 3D Cartesian coordinates to spherical (colatitude, longitude).
    
    Args:
        p_cart (np.ndarray): 3D Cartesian coordinates [x, y, z].

    Returns:
        tuple: (theta, phi) in radians.
    """
    p_cart = p_cart / np.linalg.norm(p_cart)
    theta = np.arccos(p_cart[2])
    phi = np.arctan2(p_cart[1], p_cart[0])
    return theta, phi

# --- Distance and Interpolation ---
def great_circle_distance(p1_cart, p2_cart):
    """
    Calculates the great-circle distance (angle in radians) between two points in 3D.
    
    Args:
        p1_cart (np.ndarray): Cartesian coordinates of the first point.
        p2_cart (np.ndarray): Cartesian coordinates of the second point.

    Returns:
        float: The angular distance in radians.
    """
    dot_product = np.clip(np.dot(p1_cart, p2_cart), -1.0, 1.0)
    return np.arccos(dot_product)

def slerp(p1_cart, p2_cart, t):
    """
    Spherical Linear Interpolation (SLERP).
    
    Args:
        p1_cart (np.ndarray): Cartesian start point.
        p2_cart (np.ndarray): Cartesian end point.
        t (float or np.ndarray): Interpolation factor, from 0.0 to 1.0.

    Returns:
        np.ndarray: The interpolated point(s) in Cartesian coordinates.
    """
    omega = great_circle_distance(p1_cart, p2_cart)
    if np.isclose(omega, 0):
        return p1_cart
    
    sin_omega = np.sin(omega)
    term1 = np.sin((1 - t) * omega) / sin_omega
    term2 = np.sin(t * omega) / sin_omega
    
    # Handle broadcasting for t as an array
    if hasattr(t, '__len__'):
        return term1[:, np.newaxis] * p1_cart + term2[:, np.newaxis] * p2_cart
    return term1 * p1_cart + term2 * p2_cart

# --- Tangent Space and Projections ---
def get_orthonormal_vectors(p_cart):
    """
    Gets two orthonormal vectors (a basis) for the tangent plane at a point on the sphere.
    
    Args:
        p_cart (np.ndarray): The point on the sphere in Cartesian coordinates.

    Returns:
        tuple: (tangent1, tangent2) as two np.ndarrays.
    """
    # A robust method to find a non-collinear vector
    v_aux = np.array([1.0, 0.0, 0.0])
    if np.allclose(np.abs(np.dot(p_cart, v_aux)), 1.0):
        v_aux = np.array([0.0, 1.0, 0.0])
        
    tangent1 = np.cross(p_cart, v_aux)
    tangent1 /= np.linalg.norm(tangent1)
    
    tangent2 = np.cross(p_cart, tangent1)
    tangent2 /= np.linalg.norm(tangent2)
    
    return tangent1, tangent2

def dist_point_to_arc(p_cart, arc_start_cart, arc_end_cart):
    """
    Calculates the shortest great-circle distance from a point to a great-circle arc.
    
    Args:
        p_cart (np.ndarray): The point to measure from.
        arc_start_cart (np.ndarray): The start of the arc.
        arc_end_cart (np.ndarray): The end of the arc.

    Returns:
        float: The shortest angular distance in radians.
    """
    # Normal to the plane defined by the arc
    arc_plane_normal = np.cross(arc_start_cart, arc_end_cart)
    norm = np.linalg.norm(arc_plane_normal)
    if np.isclose(norm, 0): # Start and end points are the same or opposite
        return great_circle_distance(p_cart, arc_start_cart)
    arc_plane_normal /= norm

    # Project the point onto the arc's plane
    p_proj_on_plane = p_cart - np.dot(p_cart, arc_plane_normal) * arc_plane_normal
    
    # The point might be on the opposite side of the sphere, so we find the intersection
    # with the great circle, but this is complex. A simpler way:
    # Check if the closest point on the great circle lies within the arc segment.
    
    # Vector rejection of start and end from point gives direction to closest point on GC
    v_start = arc_start_cart - np.dot(arc_start_cart, p_cart) * p_cart
    v_end = arc_end_cart - np.dot(arc_end_cart, p_cart) * p_cart
    
    # If the rejection vectors point in opposite directions, the point is between them
    if np.dot(v_start, v_end) < 0:
        # Distance to the great circle itself
        return np.abs(np.pi / 2 - great_circle_distance(p_cart, arc_plane_normal))
    else:
        # Closest point is one of the endpoints
        return min(great_circle_distance(p_cart, arc_start_cart), great_circle_distance(p_cart, arc_end_cart))
