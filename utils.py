import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def save_slice_visuals(model, dataset, epoch, criterion, device, save_dir):
    """Generates a 2D slice comparison."""
    model.eval()
    try:
        idx = np.random.randint(0, len(dataset))
        # Unpack tuple
        geo, phys, targets = dataset[idx]

        geo = geo.unsqueeze(0).to(device)
        phys = phys.unsqueeze(0).to(device)
        targets = targets.unsqueeze(0).to(device)

        with torch.no_grad():
            # Pass both inputs
            preds = model(geo, phys)
            loss = criterion(preds, targets).item()

        pts = geo[0].cpu().numpy()  # Just geometry for plotting
        pred_vals = preds[0].cpu().numpy()
        true_vals = targets[0].cpu().numpy()

        # Slice logic (Y-axis)
        y_center = (pts[:, 1].max() + pts[:, 1].min()) / 2
        slice_thickness = (pts[:, 1].max() - pts[:, 1].min()) * 0.05
        mask = np.abs(pts[:, 1] - y_center) < slice_thickness

        if np.sum(mask) < 100:
            z_center = (pts[:, 2].max() + pts[:, 2].min()) / 2
            slice_thickness = (pts[:, 2].max() - pts[:, 2].min()) * 0.05
            mask = np.abs(pts[:, 2] - z_center) < slice_thickness

        pts_slice = pts[mask]
        pred_slice = pred_vals[mask]
        true_slice = true_vals[mask]

        val_min = min(pred_slice[:, 0].min(), true_slice[:, 0].min())
        val_max = max(pred_slice[:, 0].max(), true_slice[:, 0].max())

        # Plot
        fig = plt.figure(figsize=(18, 6))

        ax1 = fig.add_subplot(131)
        ax1.scatter(pts_slice[:, 0], pts_slice[:, 2], c=pred_slice[:, 0],
                    cmap='jet', s=4, alpha=1.0, vmin=val_min, vmax=val_max)
        ax1.set_title(f"AI Prediction (Slice)\nEpoch {epoch} | Loss: {loss:.0f}")
        ax1.set_aspect('equal')
        ax1.set_facecolor('#f0f0f0')

        ax2 = fig.add_subplot(132)
        p2 = ax2.scatter(pts_slice[:, 0], pts_slice[:, 2], c=true_slice[:, 0],
                         cmap='jet', s=4, alpha=1.0, vmin=val_min, vmax=val_max)
        ax2.set_title("Ground Truth (Slice)")
        ax2.set_aspect('equal')
        ax2.set_facecolor('#f0f0f0')

        cbar = fig.colorbar(p2, ax=[ax1, ax2], fraction=0.046, pad=0.04)
        cbar.set_label('Pressure (Pa)')

        ax3 = fig.add_subplot(133)
        indices = np.random.choice(len(true_vals), 2000, replace=False)
        ax3.scatter(true_vals[indices, 0], pred_vals[indices, 0], alpha=0.3, s=2, c='purple')
        ax3.plot([val_min, val_max], [val_min, val_max], 'r--', lw=2)
        ax3.set_title('Parity Plot')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.25)

        path = os.path.join(save_dir, f"slice_epoch_{epoch}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved: {path}")

    except Exception as e:
        print(f"Warning: Visualization failed: {e}")
    model.train()

    import pyvista as pv

    def save_3d_gif(model, dataset, epoch, device, save_dir):
        """
        Creates a rotating 3D GIF of the prediction.
        """
        model.eval()
        try:
            # 1. Random Sample
            idx = np.random.randint(0, len(dataset))
            geo, phys, _ = dataset[idx]

            # Prepare for Model
            geo_dev = geo.unsqueeze(0).to(device)
            phys_dev = phys.unsqueeze(0).to(device)

            with torch.no_grad():
                preds = model(geo_dev, phys_dev)  # Output: [1, N, 1]

            pred_vals = preds[0].cpu().numpy().flatten()
            points = geo.numpy()

            # 2. PyVista Plotting
            cloud = pv.PolyData(points)
            cloud["Pressure"] = pred_vals

            plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
            plotter.add_mesh(cloud, scalars="Pressure", cmap="jet", point_size=3, render_points_as_spheres=True)
            plotter.view_iso()
            plotter.show_grid()

            # 3. Create GIF
            gif_path = os.path.join(save_dir, f"epoch_{epoch}_3d.gif")

            # Rotate camera
            path = plotter.open_gif(gif_path)
            for angle in range(0, 360, 10):  # 36 Frames
                plotter.camera.Azimuth(10)
                plotter.write_frame()

            plotter.close()
            print(f"GIF saved: {gif_path}")

        except Exception as e:
            print(f"⚠GIF creation failed: {e}")


import pyvista as pv


def save_3d_gif(model, dataset, epoch, device, save_dir):
    """
    Creates a rotating 3D GIF of the prediction.
    """
    model.eval()
    try:
        # 1. Random Sample
        idx = np.random.randint(0, len(dataset))
        geo, phys, _ = dataset[idx]

        # Prepare for Model
        geo_dev = geo.unsqueeze(0).to(device)
        phys_dev = phys.unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(geo_dev, phys_dev)  # Output: [1, N, 1]

        pred_vals = preds[0].cpu().numpy().flatten()
        points = geo.numpy()

        # 2. PyVista Plotting
        cloud = pv.PolyData(points)
        cloud["Pressure"] = pred_vals

        plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
        plotter.add_mesh(cloud, scalars="Pressure", cmap="jet", point_size=3, render_points_as_spheres=True)
        plotter.view_iso()
        plotter.show_grid()

        # 3. Create GIF
        gif_path = os.path.join(save_dir, f"epoch_{epoch}_3d.gif")

        # Rotate camera
        path = plotter.open_gif(gif_path)
        for angle in range(0, 360, 10):  # 36 Frames
            plotter.camera.Azimuth(10)
            plotter.write_frame()

        plotter.close()
        print(f"GIF saved: {gif_path}")

    except Exception as e:
        print(f"GIF creation failed: {e}")