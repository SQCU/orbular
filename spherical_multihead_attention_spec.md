# Specification: Spherical Multi-Head Attention (S-MHA)

This document specifies a layer that extends Spherical Convolutions to replicate the functionality of Multi-Head Attention, providing a mechanism for learning dynamic, long-range, and rotation-equivariant relationships between features on a sphere.

### 1. Motivation: From Static Filters to Dynamic Attention

A standard Spherical Convolution applies a single, learned filter `k` across the entire spherical input `f`. This is powerful but static; the filter is independent of the input content.

Multi-Head Attention, conversely, computes its "filter" (the attention pattern) dynamically based on the input content itself. The goal of S-MHA is to create a **dynamic spherical filter** that is computed on-the-fly from the input features, allowing the network to "spherically attend" to different parts of its own hidden states.

### 2. Core Mechanism: A Single Attention Head

Let's first define a single head. The input is a spherical feature map `f_in` (represented by its spectral coefficients `F(f_in)`).

**Step 1: Generate Q, K, and V Spherical Functions**
Instead of linear projections on token vectors, we use **spherical convolutions** to generate Query, Key, and Value *functions* on the sphere. We define three learned spherical filters: `k_Q`, `k_K`, and `k_V`.

*   **Query function:** `Q = f_in * k_Q`
*   **Key function:** `K = f_in * k_K`
*   **Value function:** `V = f_in * k_V`

These operations are performed efficiently in the spectral domain:
*   `F(Q) = F(f_in) * F(k_Q)`
*   `F(K) = F(f_in) * F(k_K)`
*   `F(V) = F(f_in) * F(k_V)`

**Step 2: Generate the Dynamic Attention Pattern**
This is the core of the analogy. In standard attention, the score matrix is `softmax(Q @ K^T)`. The matrix multiplication `Q @ K^T` computes the dot-product similarity between every pair of Q/K vectors.

On the sphere, the analogous operation to a dot product across all relative rotations is **spherical correlation**. The correlation of `Q` and `K` produces a new spherical function, `A_raw`, where the value at any point `p` represents the similarity between the `Q` function and the `K` function rotated by `p`.

*   **Raw Attention Pattern `A_raw`:** `A_raw = Correlation(Q, K)`

In the spectral domain, this correlation is a simple element-wise product:
*   `F(A_raw) = F(Q) * conj(F(K))`  (where `conj` is the complex conjugate)

To convert these similarity scores into a normalized weighting, we transform `A_raw` to the spatial domain, apply a `softmax` over all grid points, and then transform back. This creates our final attention pattern `A`.

*   **Normalized Attention Pattern `A`:** `A = ToSpectral(Softmax(ToSpatial(A_raw)))`

`A` is now a dynamically generated, input-dependent spherical filter.

**Step 3: Compute the Value-Weighted Sum**
The final output is a **spherical convolution** of the Value function `V` with the dynamic attention pattern `A`.

*   **Output `f_out`:** `f_out = V * A`

In the spectral domain, this is again an element-wise product:
*   `F(f_out) = F(V) * F(A)`

### 3. Multi-Head Extension and Output Projection

The extension to multiple heads is direct:

1.  **Parallel Heads:** The `in_channels` of the input `f_in` are split among `h` heads. The single-head process (Steps 1-3) is run in parallel for each head, each with its own learned filters (`k_Q^h`, `k_K^h`, `k_V^h`).
2.  **Concatenation:** The output feature maps from all heads, `f_out^h`, are concatenated along the channel dimension.
3.  **Output Projection:** A final spherical convolution with a `1x1` filter (`k_O`) is applied to the concatenated output. This mixes the information from the different heads and projects it back to the desired `out_channels` dimension, analogous to the `W_O` matrix in a standard transformer.

### 4. API Contract

**Name:** `SphericalMultiHeadAttention`

**Inputs:**
*   A tensor in the **spectral domain**.
*   **Shape:** `(batch_size, in_channels, l_max, m_max)`

**Outputs:**
*   A tensor in the **spectral domain**.
*   **Shape:** `(batch_size, out_channels, l_max, m_max)`

**Parameters:**
*   `in_channels` (int): Number of input channels.
*   `out_channels` (int): Number of output channels.
*   `num_heads` (int): The number of parallel attention heads.
*   **Constraint:** `in_channels` must be an integer multiple of `num_heads`.

**Internal Learnable Parameters:**
*   `h` sets of spherical filters (`k_Q^h`, `k_K^h`, `k_V^h`) for the Q, K, V convolutions.
*   One `1x1` spherical filter (`k_O`) for the final output projection.

### 5. Pseudocode

```python
def spherical_multi_head_attention(F_fin, k_Q, k_K, k_V, k_O, num_heads):
    # F_fin is the input in the spectral domain
    
    # Split channels for multi-head
    F_fin_heads = split_channels(F_fin, num_heads)
    
    head_outputs = []
    for h in range(num_heads):
        # Step 1: Generate Q, K, V in spectral domain
        F_Q_h = F_fin_heads[h] * k_Q[h]
        F_K_h = F_fin_heads[h] * k_K[h]
        F_V_h = F_fin_heads[h] * k_V[h]
        
        # Step 2: Generate Dynamic Attention Pattern
        F_A_raw_h = F_Q_h * complex_conjugate(F_K_h)
        A_raw_h_spatial = inverse_spherical_transform(F_A_raw_h)
        A_h_spatial = softmax(A_raw_h_spatial) # Softmax over spatial dimensions
        F_A_h = spherical_transform(A_h_spatial)
        
        # Step 3: Compute Value-Weighted Sum
        F_fout_h = F_V_h * F_A_h
        head_outputs.append(F_fout_h)
        
    # Concatenate head outputs and project
    F_fout_concat = concatenate_channels(head_outputs)
    F_fout_final = F_fout_concat * k_O # 1x1 convolution
    
    return F_fout_final
```
