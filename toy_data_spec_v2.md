# Toy Data Spec v2: Warped Spherical Signed Distance Fields (W-SDF)

This document specifies an advanced toy dataset designed to benchmark the capabilities of **deep** Spherical Convolutional Neural Networks. The task requires a model to learn a non-local, geometrically complex function, making it unsuitable for shallow architectures.

The problem is an extension of the Spherical Signed Distance Field (S-SDF), introducing "portals" that warp the distance metric of the sphere.

### 1. Problem Definition

The core task is to train a Spherical CNN to learn the function `F: (M, P_A, P_B) -> S_warped`, where:

*   **Input `(M, P_A, P_B)`:** A multi-channel spherical function representing:
    *   `M`: A binary mask of a "source" shape.
    *   `P_A`: A binary mask for the entry of a "portal".
    *   `P_B`: A binary mask for the exit of a "portal".
*   **Target `S_warped`:** The "Warped" Spherical Signed Distance Field. `S_warped(p)` is a scalar value representing the shortest distance from point `p` to the boundary of the source shape `M`, where the distance can be measured either as a direct great-circle path or by traveling through the portal.

### 2. Why This Requires Deep Layers

This problem is specifically designed to make shallow networks insufficient. A deep network is required for:

1.  **Non-Local Dependency:** The distance value at any given point `p` depends on the global positions of the source shape `M` and *both* portals, `P_A` and `P_B`, which can be anywhere on the sphere. The network's receptive field must be large enough to see all three objects simultaneously.
2.  **Hierarchical Reasoning:** A successful model must perform a multi-stage reasoning process that mirrors the need for deep layers:
    *   **Layer 1-2 (Feature Extraction):** Identify the distinct geometric entities: the source shape, portal A, and portal B.
    *   **Layer 3-4 (Relational Inference):** Understand the spatial relationships between these entities. The crucial piece of information to compute is the shortest distance from the exit portal `P_B` to the source shape `M`. This requires the network to "attend" to two separate hidden state representations.
    *   **Layer 5+ (Field Computation):** For every point `p` on the sphere, the network must calculate and compare two different path lengths (the direct path vs. the portal path) and select the minimum. This final, complex synthesis relies on the features and relationships derived in the earlier layers.
3.  **Spherical Attention:** The network must learn to "spherically attend" to the locations of the portals. The influence of the portals is global and directional, and the network must learn to integrate this information from its hidden states to correctly predict the final warped field.

### 3. Data Generation Process

**Step 1: Generate Shapes and Portals**
1.  **Source Shape:** Generate a primary spherical polygon `M` as described in `toy_data_spec.md`.
2.  **Portal Shapes:** Generate two additional, smaller, and non-overlapping spherical polygons, `P_A` and `P_B`, to serve as the portal entry and exit zones. Let their center points be `c_A` and `c_B`.

**Step 2: Create the Input Tensor (`M`, `P_A`, `P_B`)**
1.  **Define Grid:** Use an `L x L` equiangular grid.
2.  **Create Channels:** Create three separate binary mask channels on the grid: one for `M`, one for `P_A`, and one for `P_B`.
3.  **Stack Tensors:** Stack these three channels to form a multi-channel input tensor.

**Step 3: Create the Target Warped S-SDF (`S_warped`)**
The ground truth `S_warped` is computed for each grid point `p_grid`.

1.  **Calculate Direct Distance:** Compute `d_direct(p) = d_gc(p, boundary(M))`, the standard great-circle distance from a point `p` to the boundary of the source shape `M`.
2.  **Calculate Portal Path Distance:**
    *   First, find the shortest distance from the exit portal `P_B` to the source shape: `d_portal_to_shape = d_gc(c_B, boundary(M))`.
    *   Then, calculate the warped distance for point `p`: `d_warped(p) = d_gc(p, c_A) + d_portal_to_shape`. This is the distance from `p` to the portal entry, plus the fixed distance from the portal exit to the shape.
3.  **Calculate Final Distance:** The distance value at `p` is the minimum of the two paths: `d_final(p) = min(d_direct(p), d_warped(p))`.
4.  **Assign Sign:** The sign is determined by whether `p` is inside the original source shape `M` (negative for inside, positive for outside).
5.  **Create Tensor:** This produces the final 2D array of signed, warped distances.

### 4. Data Schema

**Input Tensor (`X`)**
*   **Name:** `shape_and_portal_masks`
*   **Shape:** `(N_batch, 3, L, L)`
*   **dtype:** `float32`
*   **Description:** A batch of 3-channel masks.
    *   `Channel 0`: Binary mask for the source shape `M`.
    *   `Channel 1`: Binary mask for the portal entry `P_A`.
    *   `Channel 2`: Binary mask for the portal exit `P_B`.

**Target Tensor (`Y`)**
*   **Name:** `warped_spherical_signed_distance_field`
*   **Shape:** `(N_batch, 1, L, L)`
*   **dtype:** `float32`
*   **Description:** The corresponding batch of ground truth W-SDFs, where distances are warped by the presence of the portals.
