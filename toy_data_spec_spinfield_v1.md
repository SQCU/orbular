# Toy Data Spec: Spin-Field v1 - "Jelly Letters"

## 1. High-Level Concept

This document specifies a toy dataset problem space designed to test and validate spin-aware Spherical CNNs. The concept, nicknamed "Jelly Letters," moves beyond static, scalar (spin-0) fields on a sphere to a dynamic physical system with internal, orientation-dependent state.

The system consists of deformable, jelly-like letters on the surface of a sphere. These letters interact with each other through anisotropic forces, causing them to deform (wiggle, squash, shear). The goal of a machine learning model would be to learn the "physics" of this system and predict its evolution over time.

This problem space is explicitly designed to make both spin-0 and spin-2 fields semantically meaningful and intrinsically coupled.

## 2. State Space Definition

The complete state of the system at any point in time `t` is described by a combination of a scalar field and a tensor field. This is analogous to a physical system's state being defined by both position and momentum.

### 2.1. Spin-0 Field: "The Jelly"
- **Physical Meaning:** Represents the presence and density of the letters themselves. It is the "position" component of the state.
- **Mathematical Representation:** A real-valued scalar field, `F_0(t)`, defined on the sphere.
- **Data Structure:** A `(resolution, resolution)` grid of floating-point values, e.g., `[0.0, 1.0]`.

### 2.2. Spin-2 Field: "The Stress"
- **Physical Meaning:** Represents the internal stress, strain, or deformation of the jelly at each point. It describes how the jelly is being squashed, stretched, or sheared. This is the "momentum" or "internal state" component.
- **Mathematical Representation:** A complex-valued, spin-2 tensor field, `F_2(t)`, defined on the sphere.
- **Data Structure:** A `(resolution, resolution)` grid of complex numbers. The magnitude and phase of each complex number encode the intensity and orientation of the local deformation.

## 3. System Dynamics ("The Physics")

A procedural generator for this dataset would simulate the following rules to generate a time series of states:

1.  **Anisotropic Repulsion:** Letters repel each other. This force is a tensor, not a simple scalar push. The repulsion force's effect (the resulting stress) depends on the relative orientation of the interacting points. For example, a point on another letter approaching from the "north" might induce a vertical squashing, while an approach from the "east" might induce a diagonal shear.
2.  **Deformation:** The stress field (`F_2`) directly influences the shape of the jelly (`F_0`). High stress in a region will cause the jelly's density to change, simulating a deformation.
3.  **Stress Propagation & Inertia:** The stress field itself evolves. It propagates across the surface of the letters and has some "memory" or inertia, causing the jelly to oscillate or wiggle back after being deformed.
4.  **Equivariance:** The entire system is subject to global SO(3) rotations. The rules of physics must be equivariant to these rotations.

## 4. Machine Learning Task

-   **Goal:** To predict the full state of the system at timestep `t+1` given the full state at timestep `t`.
-   **Input:** The complete state `S(t) = { F_0(t), F_2(t) }`. This would be fed into the network as a multi-channel, multi-spin input.
-   **Output (Prediction):** The predicted state `S_pred(t+1) = { F_0_pred(t+1), F_2_pred(t+1) }`.
-   **Ground Truth:** The actual, simulated state `S_true(t+1)`.

## 5. Loss Function

The loss function must penalize errors in both the predicted shape and the predicted internal stress.

-   **Composite Loss:** A weighted sum of the Mean Squared Error for each field.
    `Loss_total = w_0 * MSE(F_0_pred, F_0_true) + w_2 * Loss_complex(F_2_pred, F_2_true)`

-   **Spin-0 Loss (`MSE`):** A standard mean squared error on the real-valued scalar fields.
-   **Spin-2 Loss (`Loss_complex`):** The mean squared error of the difference between the complex fields: `mean(|F_2_pred - F_2_true|^2)`. This single term correctly penalizes errors in both the magnitude and the orientation (phase) of the predicted stress field.

The weights `w_0` and `w_2` can be tuned to adjust the relative importance of getting the shape correct versus getting the internal physics correct.
