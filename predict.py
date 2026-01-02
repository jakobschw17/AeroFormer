import argparse
import torch
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import os
from model import ProductionTransformer
from dataset import parse_info_file

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_comparison_image(points, pred_p, true_p, save_path):
    """
    Generates a high-quality comparison image (Slice + Parity Plot).
    """
    # 1. Slice Logic (Cut through the middle of the car)
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    y_center = (y_max + y_min) / 2
    slice_thickness = (y_max - y_min) * 0.05

    mask = np.abs(points[:, 1] - y_center) < slice_thickness

    # Fallback to Z-slice if Y-slice is empty
    if np.sum(mask) < 100:
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        z_center = (z_max + z_min) / 2
        slice_thickness = (z_max - z_min) * 0.05
        mask = np.abs(points[:, 2] - z_center) < slice_thickness

    pts_slice = points[mask]
    pred_slice = pred_p[mask]

    # Handle Ground Truth
    has_gt = true_p is not None
    if has_gt:
        true_slice = true_p[mask]
        val_min = min(pred_slice.min(), true_slice.min())
        val_max = max(pred_slice.max(), true_slice.max())
    else:
        val_min = pred_slice.min()
        val_max = pred_slice.max()

    # 2. Plotting
    fig = plt.figure(figsize=(18, 6))

    # -- Subplot 1: AI Prediction --
    ax1 = fig.add_subplot(131)
    sc1 = ax1.scatter(pts_slice[:, 0], pts_slice[:, 2], c=pred_slice,
                      cmap='jet', s=4, alpha=1.0, vmin=val_min, vmax=val_max)
    ax1.set_title("AI Prediction (Center Slice)")
    ax1.set_aspect('equal')
    ax1.set_facecolor('#f0f0f0')

    # -- Subplot 2: Ground Truth --
    ax2 = fig.add_subplot(132)
    if has_gt:
        sc2 = ax2.scatter(pts_slice[:, 0], pts_slice[:, 2], c=true_slice,
                          cmap='jet', s=4, alpha=1.0, vmin=val_min, vmax=val_max)
        ax2.set_title("Ground Truth (CFD)")
    else:
        ax2.text(0.5, 0.5, "No Ground Truth\nAvailable", ha='center', va='center')
        ax2.set_title("Ground Truth")

    ax2.set_aspect('equal')
    ax2.set_facecolor('#f0f0f0')

    # Colorbar
    cbar = fig.colorbar(sc1, ax=[ax1, ax2], fraction=0.046, pad=0.04)
    cbar.set_label('Pressure (Pa)')

    # -- Subplot 3: Parity Plot --
    ax3 = fig.add_subplot(133)
    if has_gt:
        indices = np.random.choice(len(true_p), min(2000, len(true_p)), replace=False)
        ax3.scatter(true_p[indices], pred_p[indices], alpha=0.3, s=2, c='purple')
        ax3.plot([val_min, val_max], [val_min, val_max], 'r--', lw=2)
        ax3.set_xlabel("True Pressure")
        ax3.set_ylabel("Predicted Pressure")
        ax3.set_title('Parity Plot')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No Data for Validation", ha='center', va='center')

    # Removed tight_layout() to fix warning, bbox_inches='tight' handles it
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"🖼️  Visualization saved: {save_path}")


def save_rotating_gif(mesh, pred_vals, save_path):
    """
    Creates a rotating 3D GIF of the prediction using PyVista.
    """
    try:
        print("🎥 Generating 3D GIF (this might take a few seconds)...")
        cloud = mesh.copy()
        cloud["Pressure_Pred"] = pred_vals

        plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
        plotter.add_mesh(cloud, scalars="Pressure_Pred", cmap="jet",
                         point_size=3, render_points_as_spheres=True)

        # --- FIX: Use view_isometric() instead of view_iso() ---
        plotter.view_isometric()
        plotter.show_grid()

        # Open GIF
        plotter.open_gif(save_path)

        # Rotate 360 degrees
        frames = 36
        for angle in range(frames):
            plotter.camera.Azimuth(360 // frames)
            plotter.write_frame()

        plotter.close()
        print(f"🎞️  GIF saved: {save_path}")
    except Exception as e:
        print(f"Warning: GIF creation failed. {e}")


def predict(vtp_path, info_path, model_path):
    print(f"🔍 Loading model from {model_path}...")

    checkpoint = torch.load(model_path, map_location=device)

    # Init Model (num_layers=6 to match training)
    model = ProductionTransformer(geo_dim=3, phys_dim=7, output_dim=4, num_layers=6).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Prepare Data
    print(f"📦 Reading geometry: {os.path.basename(vtp_path)}")
    mesh = pv.read(vtp_path)
    pos = mesh.points.astype(np.float32)
    pos_centered = pos - np.mean(pos, axis=0)

    phys_feats = parse_info_file(info_path)

    # To Tensor
    geo_tensor = torch.from_numpy(pos_centered).unsqueeze(0).to(device)
    phys_tensor = torch.from_numpy(phys_feats).unsqueeze(0).to(device)

    # Predict
    print("⚡ Running inference...")
    with torch.no_grad():
        preds = model(geo_tensor, phys_tensor)

    pred_data = preds[0].cpu().numpy()
    pred_pressure = pred_data[:, 0]
    pred_velocity = pred_data[:, 1:]

    # --- 1. Save VTP ---
    mesh["Pred_Pressure"] = pred_pressure
    mesh["Pred_Velocity"] = pred_velocity
    out_vtp = "prediction_result.vtp"
    mesh.save(out_vtp)
    print(f"✅ Saved 3D mesh: {out_vtp}")

    # --- 2. Extract Ground Truth ---
    true_pressure = None
    if 'p' in mesh.point_data:
        true_pressure = mesh.point_data['p']
        print("✅ Ground Truth found. Generating comparison...")
    else:
        print("ℹ️  No Ground Truth found. Generating prediction-only visuals...")

    # --- 3. Generate Static Image ---
    save_comparison_image(
        points=pos,
        pred_p=pred_pressure,
        true_p=true_pressure,
        save_path="prediction_slice.png"
    )

    # --- 4. Generate Rotating GIF ---
    save_rotating_gif(
        mesh=mesh,
        pred_vals=pred_pressure,
        save_path="prediction_3d.gif"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, required=True, help="Path to .vtp file")
    parser.add_argument("--info", type=str, required=True, help="Path to .txt info file")
    parser.add_argument("--model", type=str, required=True, help="Path to .pth checkpoint")
    args = parser.parse_args()

    predict(args.mesh, args.info, args.model)