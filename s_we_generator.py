import numpy as np
import matplotlib.pyplot as plt

# 1. --- Vector Font Definition ---
# A simple font defining characters as a list of line segments in a [-1, 1] box.
# Each line segment is a tuple of (start_point, end_point).
VECTOR_FONT = {
    'H': [((-0.8, 1), (-0.8, -1)), ((0.8, 1), (0.8, -1)), ((-0.8, 0), (0.8, 0))],
    'E': [((-0.8, 1), (-0.8, -1)), ((-0.8, 1), (0.8, 1)), ((-0.8, 0), (0.5, 0)), ((-0.8, -1), (0.8, -1))],
    'L': [((-0.8, 1), (-0.8, -1)), ((-0.8, -1), (0.8, -1))],
    'O': [((-0.8, 1), (0.8, 1)), ((0.8, 1), (0.8, -1)), ((0.8, -1), (-0.8, -1)), ((-0.8, -1), (-0.8, 1))],
    'W': [((-0.8, 1), (-0.6, -1)), ((-0.6, -1), (0, 0)), ((0, 0), (0.6, -1)), ((0.6, -1), (0.8, 1))],
    'R': [((-0.8, 1), (-0.8, -1)), ((-0.8, 1), (0.5, 1)), ((0.5, 1), (0.5, 0)), ((0.5, 0), (-0.8, 0)), ((-0.2, 0), (0.6, -1))],
    'D': [((-0.8, 1), (-0.8, -1)), ((-0.8, 1), (0.3, 1)), ((0.3, 1), (0.8, 0.5)), ((0.8, 0.5), (0.8, -0.5)), ((0.8, -0.5), (0.3, -1)), ((0.3, -1), (-0.8, -1))],
    'S': [((0.8, 1), (-0.8, 1)), ((-0.8, 1), (-0.8, 0)), ((-0.8, 0), (0.8, 0)), ((0.8, 0), (0.8, -1)), ((0.8, -1), (-0.8, -1))],
    'F': [((-0.8, 1), (-0.8, -1)), ((-0.8, 1), (0.8, 1)), ((-0.8, 0), (0.5, 0))]
}

# 2. --- Core Geometry Functions ---
def spherical_to_cartesian(theta, phi):
    """Converts spherical coordinates (colatitude, longitude) to 3D Cartesian."""
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])

def great_circle_distance(p1_cart, p2_cart):
    """Calculates the great-circle distance (angle in radians) between two points in 3D."""
    dot_product = np.clip(np.dot(p1_cart, p2_cart), -1.0, 1.0)
    return np.arccos(dot_product)

def slerp(p1_cart, p2_cart, t):
    """Spherical Linear Interpolation."""
    omega = great_circle_distance(p1_cart, p2_cart)
    if omega == 0:
        return p1_cart
    term1 = np.sin((1 - t) * omega) / np.sin(omega) * p1_cart
    term2 = np.sin(t * omega) / np.sin(omega) * p2_cart
    return term1 + term2

def get_orthonormal_vectors(p_cart):
    """Gets two orthonormal vectors tangent to the sphere at point p_cart."""
    # Simplified approach: find a non-collinear vector and use cross products
    if np.allclose(p_cart, [0, 0, 1]):
        v_aux = np.array([1, 0, 0])
    else:
        v_aux = np.array([0, 0, 1])
    
    tangent1 = np.cross(p_cart, v_aux)
    tangent1 /= np.linalg.norm(tangent1)
    
    tangent2 = np.cross(p_cart, tangent1)
    tangent2 /= np.linalg.norm(tangent2)
    
    return tangent1, tangent2

def dist_point_to_arc(p_cart, arc_start_cart, arc_end_cart):
    """Calculates the shortest great-circle distance from a point to a great-circle arc."""
    arc_normal = np.cross(arc_start_cart, arc_end_cart)
    arc_normal /= np.linalg.norm(arc_normal)
    
    # Distance from point to the great circle containing the arc
    dist_to_gc = np.abs(np.pi / 2 - great_circle_distance(p_cart, arc_normal))
    
    # Project point onto the great circle
    p_proj = p_cart - np.dot(p_cart, arc_normal) * arc_normal
    p_proj /= np.linalg.norm(p_proj)
    
    # Check if the projection lies on the arc
    on_arc = great_circle_distance(arc_start_cart, p_proj) <= great_circle_distance(arc_start_cart, arc_end_cart) and \
             great_circle_distance(arc_end_cart, p_proj) <= great_circle_distance(arc_start_cart, arc_end_cart)
             
    if on_arc:
        return dist_to_gc
    else:
        # If not on the arc, the closest point is one of the endpoints
        return min(great_circle_distance(p_cart, arc_start_cart), great_circle_distance(p_cart, arc_end_cart))

# 3. --- S-SDF and Mask Generation ---
def generate_word_sdf(word, grid_size=128, path_start_sph=(np.pi/4, np.pi/4), path_end_sph=(np.pi/2, 3*np.pi/2), type_size_rad=0.2):
    """
    Generates the binary mask and the ground truth SDF for a word embossed on a sphere.
    """
    # Create the spherical grid
    theta = np.linspace(0, np.pi, grid_size)
    phi = np.linspace(0, 2 * np.pi, grid_size)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    # Convert grid to Cartesian coordinates for distance calculations
    grid_cart = spherical_to_cartesian(theta_grid.ravel(), phi_grid.ravel()).T
    
    # --- Generate all stroke geometries for the word ---
    all_strokes_cart = []
    path_start_cart = spherical_to_cartesian(*path_start_sph)
    path_end_cart = spherical_to_cartesian(*path_end_sph)
    
    num_letters = len(word)
    for i, char in enumerate(word):
        if char.upper() not in VECTOR_FONT:
            continue
            
        # Find the letter's center point on the great-circle path
        t = (i + 0.5) / num_letters
        letter_center_cart = slerp(path_start_cart, path_end_cart, t)
        
        # Get local coordinate system on the sphere's surface
        u_vec, v_vec = get_orthonormal_vectors(letter_center_cart)
        
        # Map 2D font strokes to the sphere
        for p1_2d, p2_2d in VECTOR_FONT[char.upper()]:
            stroke_start_cart = letter_center_cart + (p1_2d[0] * u_vec + p1_2d[1] * v_vec) * (type_size_rad / 2)
            stroke_end_cart = letter_center_cart + (p2_2d[0] * u_vec + p2_2d[1] * v_vec) * (type_size_rad / 2)
            
            stroke_start_cart /= np.linalg.norm(stroke_start_cart)
            stroke_end_cart /= np.linalg.norm(stroke_end_cart)
            
            all_strokes_cart.append((stroke_start_cart, stroke_end_cart))

    # --- Calculate SDF for each grid point ---
    sdf = np.full(grid_cart.shape[0], np.inf)
    for i, p_cart in enumerate(grid_cart):
        min_dist = np.inf
        for arc_start, arc_end in all_strokes_cart:
            dist = dist_point_to_arc(p_cart, arc_start, arc_end)
            if dist < min_dist:
                min_dist = dist
        sdf[i] = min_dist
        
    sdf_grid = sdf.reshape((grid_size, grid_size)).T
    
    # --- Generate binary mask from SDF ---
    stroke_thickness_rad = type_size_rad * 0.1
    mask_grid = (sdf_grid < stroke_thickness_rad).astype(np.float32)
    
    return mask_grid, sdf_grid

# 4. --- SDF Decoder ---
def decode_sdf_to_texture(sdf_tensor, threshold=0.02):
    """
    Converts an SDF tensor back to a binary texture for visualization.
    """
    return (sdf_tensor < threshold).astype(np.float32)

# 5. --- Visualization ---
if __name__ == '__main__':
    WORD_TO_RENDER = "HELLO"
    GRID_RESOLUTION = 256
    
    print(f"Generating S-SDF for the word: '{WORD_TO_RENDER}'...")
    
    # Generate the data
    input_mask, target_sdf = generate_word_sdf(
        WORD_TO_RENDER, 
        grid_size=GRID_RESOLUTION,
        path_start_sph=(np.pi/3, np.pi/4),
        path_end_sph=(2*np.pi/3, 5*np.pi/3),
        type_size_rad=0.3
    )
    
    # Decode the ground truth SDF for comparison
    decoded_texture = decode_sdf_to_texture(target_sdf, threshold=0.03)
    
    print("Generation complete. Plotting results...")
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Spherical Word Embossing: '{WORD_TO_RENDER}'", fontsize=16)
    
    # Plot Input Mask
    ax1 = axes[0]
    im1 = ax1.imshow(input_mask, cmap='gray_r', extent=[0, 360, 0, 180])
    ax1.set_title("Input: Binary Mask (M)")
    ax1.set_xlabel("Longitude (degrees)")
    ax1.set_ylabel("Latitude (degrees)")
    
    # Plot Target SDF
    ax2 = axes[1]
    im2 = ax2.imshow(target_sdf, cmap='hot', extent=[0, 360, 0, 180])
    ax2.set_title("Target: Ground Truth S-SDF (S)")
    ax2.set_xlabel("Longitude (degrees)")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Plot Decoded Texture
    ax3 = axes[2]
    im3 = ax3.imshow(decoded_texture, cmap='gray_r', extent=[0, 360, 0, 180])
    ax3.set_title("Decoded Texture from S-SDF")
    ax3.set_xlabel("Longitude (degrees)")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_filename = "spherical_word_embossing_visualization.png"
    plt.savefig(output_filename)
    
    print(f"Visualization saved to '{output_filename}'")
    plt.show()
