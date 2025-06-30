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

### Spherical CNN Demo

This feature demonstrates a complete end-to-end test of the S-SDF pipeline with a minimal, randomly initialized Spherical CNN.

#### Sourcemap

The entry point for this feature is `run_model_demo.py`. Here is the high-level call graph:

1.  **`run_model_demo.main()`** - Orchestrates the data generation, model execution, and visualization.
    *   Calls `ssdf_encoder_optimized.encode_to_sdf_and_mask_optimized()` to generate the input mask and target S-SDF.
    *   Instantiates the `SimpleSphericalCNN` model from `spherical_cnn.py`.
    *   Calls `torch.from_numpy()` to convert the data into PyTorch tensors.
    *   Performs the forward pass: `model(input_tensor)`.

2.  **`spherical_cnn.SimpleSphericalCNN.forward()`** - The model's forward pass.
    *   `nn.Conv2d` (Embedder): Lifts the input from 1 to `intermediate_channels`.
    *   `spherical_cnn.SphericalConv2d.forward()`: Called three times in sequence.
        *   `s2fft.forward()`: Transforms the input tensor from the spatial to the harmonic domain (iterating over channels).
        *   `torch.einsum()`: Performs the convolution by multiplying the input and weights in the harmonic domain.
        *   `s2fft.inverse()`: Transforms the result back to the spatial domain (iterating over channels).
    *   `nn.Conv2d` (Unembedder): Projects the features back to a single-channel output.

3.  **Back in `run_model_demo.main()`**:
    *   `torch.nn.functional.interpolate()`: Upsamples the model's output to match the target's resolution.
    *   `ssdf_loss.mean_squared_error_sdf()`: Calculates the loss between the upsampled model output and the target.
    *   `matplotlib.pyplot`: Plots the input, target, raw model output, and the final error surface.

### Spherical Attention Demo

This feature demonstrates the `SphericalMultiHeadAttention` layer.

#### Sourcemap

The entry point for this feature is `run_sphereattn_demo.py`. Here is the high-level call graph:

1.  **`run_sphereattn_demo.main()`** - Orchestrates the data generation, model execution, and visualization.
    *   Calls `generate_sdf_for_path()` to generate the target S-SDF.
    *   Instantiates the `SphericalMultiHeadAttention` model from `spherical_attention.py`.
    *   Performs the forward pass: `model(random_input_lm)`.
    *   `ssdf_loss.mean_squared_error_sdf()`: Calculates the loss between the model output and the target.
    *   `matplotlib.pyplot`: Plots the target, prediction, and loss in both the SDF and spherical texture domains.

