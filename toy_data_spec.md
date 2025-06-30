# Toy Data Spec: Spherical Signed Distance Fields (S-SDF)

This document outlines a process for generating a toy dataset suitable for training and evaluating Spherical Convolutional Neural Networks. The task is to learn a mapping from a binary mask of a 2D shape on a sphere's surface to its corresponding Spherical Signed Distance Field (S-SDF).

This provides a more visual and geometric problem compared to tasks like molecular property prediction, making it ideal for debugging, visualization, and intuitive understanding of a model's capabilities.

### 1. Problem Definition

The core task is to train a Spherical CNN to approximate the function `F: M -> S`, where:

*   **`M` is the Input:** A binary mask representing a simple shape on the surface of a unit sphere. It is a spherical function where `M(p) = 1` if point `p` is inside the shape, and `0` otherwise.
*   **`S` is the Target:** The corresponding Spherical Signed Distance Field. `S(p)` is a scalar value representing the shortest great-circle distance from point `p` to the boundary of the shape. The sign is negative if `p` is inside the shape and positive if it is outside.

A model that learns this mapping demonstrates an understanding of spherical geometry and the concept of distance on a curved manifold.

### 2. Data Generation Process

A single data sample (an `(M, S)` pair) is generated as follows:

**Step 1: Generate a Random Spherical Polygon**
A simple, closed shape is defined as a spherical polygon.

1.  **Choose a Pole:** A random point `P_pole` is selected on the sphere's surface. This ensures the polygon does not cover the entire sphere.
2.  **Sample Vertices:** A number of vertices, `N` (e.g., randomly from 3 to 8), are sampled on the hemisphere centered at `P_pole`.
3.  **Order Vertices:** The vertices are ordered by their angle around the pole `P_pole` to form a simple (non-self-intersecting) polygon.
4.  **Define Edges:** The edges of the polygon are the **great-circle arcs** connecting consecutive vertices.

**Step 2: Create the Input Binary Mask (`M`)**
The input is a discretized representation of the polygon on a spherical grid.

1.  **Define Grid:** An `L x L` equiangular grid (e.g., `64x64`) is defined over the sphere, with coordinates `(θ, φ)`.
2.  **Point-in-Polygon Test:** For each grid point `p_grid`, determine if it lies inside or outside the spherical polygon. This can be done by summing the angles subtended by each edge from the point `p_grid`. If the sum is `2π`, the point is inside; if `0`, it's outside.
3.  **Create Tensor:** This produces a 2D array of `0`s and `1`s, which forms the single channel of the input tensor.

**Step 3: Create the Target S-SDF (`S`)**
The target is the ground truth S-SDF, also discretized on the same grid.

1.  **Calculate Distance to Boundary:** For each grid point `p_grid`, calculate its shortest great-circle distance to the boundary of the polygon. This is the minimum of the great-circle distances from `p_grid` to each of the polygon's edges.
2.  **Assign Sign:** The sign of the distance is determined by the point-in-polygon test from Step 2 (negative for inside, positive for outside).
3.  **Create Tensor:** This produces a 2D array of signed, real-valued distances, which forms the single channel of the target tensor. The distance is measured in radians (from `[-π, π]`).

### 3. Data Schema

**Input Tensor (`X`)**
*   **Name:** `shape_binary_mask`
*   **Shape:** `(N_batch, 1, L, L)`
*   **dtype:** `float32`
*   **Description:** A batch of binary masks on a spherical grid. `1.0` represents points inside the shape, `0.0` represents points outside.

**Target Tensor (`Y`)**
*   **Name:** `spherical_signed_distance_field`
*   **Shape:** `(N_batch, 1, L, L)`
*   **dtype:** `float32`
*   **Description:** The corresponding batch of ground truth S-SDFs. Values are the great-circle distance in radians, with a negative sign for interior points.

### 4. Modeling & Evaluation

*   **Architecture:** This dataset is well-suited for a U-Net-like architecture composed of the `SphericalConvBlock`s defined in `so3_conv_spec.md`. The network would take the binary mask as input and regress the S-SDF.
*   **Loss Function:** A standard regression loss like Mean Squared Error (MSE) or Mean Absolute Error (MAE) between the predicted S-SDF and the ground truth `Y` would be appropriate.
*   **Evaluation:** The model's performance can be evaluated quantitatively with the loss function and qualitatively by visualizing the predicted S-SDF. A successful model will produce a smooth gradient field that accurately represents the distance to the input shape's boundary.
