# 🏎️ AeroFormer: Physics-Informed Point Transformer

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **A Geometry-Aware Transformer model that predicts aerodynamic pressure and velocity fields directly from 3D point clouds.**

This repository contains a PyTorch implementation of a **Point Transformer** designed to act as a surrogate model for CFD (Computational Fluid Dynamics) simulations. It takes the 3D geometry of a car (Ahmed Body) along with physical parameters (velocity, dimensions) and predicts the surface pressure and flow velocity fields in real-time.

---

## 🖼️ Gallery

![Training Demo](visuals_production/slice_epoch_740.png)
*(Example output showing Ground Truth vs. AI Prediction of pressure distribution)*

---

## 🧠 Architecture & Methodology

Unlike standard PointNets, this model utilizes a **Hybrid Tokenization Strategy**:

1.  **Geometry Stream:** The 3D point cloud (x, y, z) is embedded into point tokens.
2.  **Physics Stream:** Global scalar parameters (Velocity, Slant Angle, Ground Clearance, etc.) are projected into a single **Global Context Token**.
3.  **Self-Attention:** The Global Token is prepended to the sequence. Through self-attention, every geometric point can "attend" to the physical conditions (e.g., "How fast am I moving?"), allowing for physics-aware predictions without voxelization.

**Input:** $N \times 3$ (Point Cloud) + $1 \times 7$ (Physics Vector)  
**Output:** $N \times 4$ (Pressure $p$, Velocity $u_x, u_y, u_z$)

---

## 📂 Project Structure

```text
.
├── dataset.py          # Data loading, parsing physics info, and mesh handling
├── model.py            # Point Transformer architecture with Global Physics Token
├── train.py            # Main training loop, checkpointing, and configuration
├── utils.py            # Visualization tools (Slices, Parity Plots, GIFs)
├── predict.py          # Inference script for testing on new files
├── requirements.txt    # Python dependencies
└── visuals_production/ # Generated training progress images