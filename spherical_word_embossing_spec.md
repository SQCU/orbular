# Data Spec: Spherical Word Embossing (S-WE)

This document specifies a process for generating a dataset of "embossed" words on a sphere's surface. The goal is to train a Spherical CNN to learn the Signed Distance Field (SDF) corresponding to the geometry of the text. This task is designed to be visually intuitive, procedurally infinite, and to require hierarchical feature extraction, making it an excellent benchmark for deep spherical networks.

### 1. Problem Definition

The core task is to train a Spherical CNN to learn the function `F: M -> S`, where:

*   **`M` is the Input:** A binary mask representing a word rendered onto the surface of a unit sphere.
*   **`S` is the Target:** The corresponding high-fidelity Spherical Signed Distance Field (S-SDF). `S(p)` is a scalar value representing the shortest great-circle distance from point `p` to the centerline of the text strokes.

A model that successfully learns this mapping must recognize low-level features (line segments, curves), assemble them into higher-level concepts (letters), and understand their spatial arrangement along a path (words).

### 2. Data Generation Process

A single data sample (an `(M, S)` pair) is generated as follows:

**Step 1: Define a Simplified Vector Font**
Standard font files are too complex. We first define a simple, stroke-based vector representation for each character in the alphabet.

*   **Representation:** Each character is a set of 2D line segments and/or curves (e.g., Bezier curves) defined within a local `[-1, 1] x [-1, 1]` coordinate system.
*   **Example ('H'):** Defined by three line segments: `[(-0.5, 1), (-0.5, -1)]`, `[(0.5, 1), (0.5, -1)]`, and `[(-0.5, 0), (0.5, 0)]`.

**Step 2: Define a Spherical Path**
The word will be laid out along a parameterized path on the sphere.

*   **Path Type:** Choose a random path type, such as a great-circle arc, a spiral (loxodrome), or a spherical ellipse.
*   **Parameterization:** The path is a function `path(t) = (θ, φ)` where `t` is the normalized distance along the path, from `0` to `1`. The path is given a random start point, direction, and total arc length.

**Step 3: "Emboss" the Word onto the Sphere**
1.  **Select Word:** Randomly pick a 5-letter word from a dictionary (e.g., "HELLO").
2.  **Place Letters:** For each character in the word, determine its position `p_i = path(t_i)` along the spherical path.
3.  **Render Strokes:** At each letter's position `p_i`, create a local tangent plane. Map the character's 2D vector strokes from the font definition onto this plane.
    *   The strokes are scaled by a random "angular type size" (e.g., 5-10 degrees in height).
    *   The orientation of the letter is kept orthogonal to the path's tangent vector at that point.
4.  **Collect Geometries:** The final result is a collection of all the great-circle arcs and curves that form the letters of the word on the sphere's surface.

**Step 4: Generate Input Mask (`M`) and Target SDF (`S`)**
1.  **Define Grid:** An `L x L` equiangular grid is defined over the sphere.
2.  **Generate Target SDF (`S`):** For each grid point `p_grid`, calculate the shortest great-circle distance to any of the stroke geometries collected in Step 3. This produces a 2D array of positive, real-valued distances. This is the ground truth `S`.
3.  **Generate Input Mask (`M`):** Create the input mask by applying a threshold to the high-fidelity SDF. `M(p) = 1` if `S(p) < epsilon` (where `epsilon` is a small value defining the "thickness" of the strokes), and `0` otherwise. This simulates a lower-quality, rasterized input.

### 4. Data Schema

**Input Tensor (`X`)**
*   **Name:** `word_binary_mask`
*   **Shape:** `(N_batch, 1, L, L)`
*   **dtype:** `float32`
*   **Description:** A batch of low-fidelity binary masks of words on a spherical grid.

**Target Tensor (`Y`)**
*   **Name:** `word_signed_distance_field`
*   **Shape:** `(N_batch, 1, L, L)`
*   **dtype:** `float32`
*   **Description:** The corresponding batch of ground truth, high-fidelity S-SDFs representing the distance to the text.

### 5. SDF-to-Spheretexture Decoder (for Evaluation)

To visually evaluate the model's performance, a decoder function is required to convert a predicted SDF back into a human-readable image.

**Function: `decode_sdf_to_texture(sdf_tensor, threshold)`**

*   **Input `sdf_tensor`:** A single S-SDF, with shape `(1, L, L)`. This can be the ground truth or a model's prediction.
*   **Input `threshold`:** A small float value that determines the thickness of the reconstructed strokes.
*   **Process:**
    1.  Create an output tensor `texture` of the same `(L, L)` size, initialized to zeros.
    2.  For each point `p`, set `texture(p) = 1.0` if `sdf_tensor(p) < threshold`.
    3.  Otherwise, `texture(p) = 0.0`.
*   **Output:** A binary `(L, L)` spheretexture that can be plotted. A good model prediction will result in a texture that clearly shows the original word.
