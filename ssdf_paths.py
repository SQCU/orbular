# gemini-2.5 first pass code, under review
"""
ssdf_paths.py

Pure functions for generating parameterized paths on a sphere's surface.
These functions return a callable that maps a normalized distance `t` (from 0 to 1)
to a 3D Cartesian coordinate on the unit sphere.
"""
import numpy as np
import ssdf_geometry as geom

def great_circle_arc(start_cart, end_cart):
    """
    Returns a function that parameterizes a great-circle arc between two points.

    Args:
        start_cart (np.ndarray): Cartesian coordinates of the start point.
        end_cart (np.ndarray): Cartesian coordinates of the end point.

    Returns:
        callable: A function `path(t)` that returns the Cartesian point for `t` in [0, 1].
    """
    start_cart = np.asarray(start_cart)
    end_cart = np.asarray(end_cart)
    
    def path(t):
        """ SLERP between the start and end points. """
        return geom.slerp(start_cart, end_cart, t)
        
    return path

# --- Stub for a more complex path ---
def loxodrome_spiral(start_sph, bearing_rad, length_rad):
    """
    STUB: Generates a loxodrome (a path of constant bearing).

    This function would calculate the spiral path starting at a given point,
    with a constant angle relative to the meridians, for a certain total length.

    Args:
        start_sph (tuple): (theta, phi) start coordinates in radians.
        bearing_rad (float): The constant bearing (angle from North) in radians.
        length_rad (float): The total angular length of the spiral path.

    Returns:
        callable: A function `path(t)` that returns the Cartesian point for `t` in [0, 1].
    """
    # To be implemented:
    # 1. Convert start_sph to Cartesian.
    # 2. Define the loxodrome equation in spherical coordinates.
    # 3. Create a function that maps t -> (theta, phi) along the path.
    # 4. Convert the result to Cartesian coordinates.
    print("STUB: loxodrome_spiral function is not yet implemented.")
    
    # For now, return a simple great circle as a fallback.
    fallback_end_cart = geom.spherical_to_cartesian(start_sph[0] + length_rad, start_sph[1] + length_rad)
    return great_circle_arc(geom.spherical_to_cartesian(*start_sph), fallback_end_cart)
