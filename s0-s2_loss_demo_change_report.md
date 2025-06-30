# S0-S2 Loss Demo Change Report

This document outlines the differences between the `s0_s2_pipeline.py` script and the established standards for data generation and visualization set by the earlier demo scripts (`jelly_letter_demo.py` and `run_sphereattn_demo.py`).

## 1. Superfluous Features

*   **Whole-Sphere Visualization:** The `plot_reconstruction` function in `s0_s2_pipeline.py` currently renders the entire sphere, which obscures the extruded text and makes the visualization useless. The standard set by `run_sphereattn_demo.py` is to render *only* the extruded text against a transparent background.

## 2. Wrong Features

*   **Path Geometry:** The `s0_s2_pipeline.py` script uses a single, continuous great-circle path for the input text. This is incorrect. The standard set by `jelly_letter_demo.py` is to generate multiple, discrete paths at different locations on the sphere. This creates a more complex and realistic input for the model.

## 3. Missing Features

*   **Advanced 3D Visualization:** The `s0_s2_pipeline.py` script is missing the advanced 3D visualization logic from `run_sphereattn_demo.py`. This logic is responsible for rendering the extruded text without the sphere, which is critical for clear and interpretable visualizations.
*   **Multi-Path Generation:** The `s0_s2_pipeline.py` script is missing the multi-path generation logic from `jelly_letter_demo.py`. This logic is responsible for creating the multiple, discrete text locations that are used in the other demos.
