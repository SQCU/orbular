# Project Orbular

This project uses `uv` for managing Python dependencies in a virtual environment.

## Setup

1.  **Create the virtual environment:**
    ```bash
    uv venv
    ```

2.  **Activate the environment:**
    ```bash
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```

## Features

### Spherical Signed Distance Field (S-SDF) Generation

This project includes a modular and optimized framework for generating S-SDFs from text embossed on a sphere. This is useful for creating procedurally generated training data for spherical deep learning models.

#### Sourcemap

The primary entry point for this feature is `loss_visualization_demo_optimized.py`. Here is the high-level call graph showing how the different modules interact to generate a single sample and visualize the loss between a reference and a translated version:

1.  **`loss_visualization_demo_optimized.main()`** - Orchestrates the entire process.
    *   Calls `generate_sdf_for_path()` twice (once for the reference, once for the translated version).

2.  **`loss_visualization_demo_optimized.generate_sdf_for_path()`** - A helper to generate one complete S-SDF.
    *   `ssdf_geometry.spherical_to_cartesian()`: Converts spherical start/end points to 3D coordinates.
    *   `ssdf_paths.great_circle_arc()`: Creates a callable path function from the 3D points.
    *   `ssdf_encoder_optimized.generate_stroke_geometries()`: Generates all 3D line segments for the text along the path.
        *   `ssdf_font.VECTOR_FONT`: Looks up character definitions.
        *   `ssdf_geometry.get_orthonormal_vectors()`: Creates a local tangent plane for each character.
    *   `ssdf_encoder_optimized.encode_to_sdf_and_mask_optimized()`: The core, high-performance encoding function.
        *   `ssdf_geometry.spherical_to_cartesian()`: Creates the full spherical grid.
        *   `tqdm.tqdm()`: Wraps the stroke loop for a progress bar.
        *   `ssdf_encoder_optimized.dist_grid_to_arc_vectorized()`: (Called for each stroke) Calculates the distance from all grid points to one stroke using vectorized NumPy.

3.  **Back in `loss_visualization_demo_optimized.main()`**:
    *   `ssdf_loss.mean_squared_error_sdf()`: Calculates the scalar loss and the plottable error surface between the two generated SDFs.
    *   `ssdf_decoder.decode_sdf_to_texture()`: Converts the SDFs back into binary images for visualization.
    *   `matplotlib.pyplot`: Used to plot the final comparison of the textures and the error surface.
