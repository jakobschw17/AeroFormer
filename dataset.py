import os
import glob
import re
import numpy as np
import torch
import pyvista as pv
from torch.utils.data import Dataset
from tqdm import tqdm


def parse_info_file(filepath):
    """Parses physical parameters and normalizes them."""
    data = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.split(':')
                if len(parts) == 2:
                    data[parts[0].strip()] = float(parts[1].strip())

        # Normalize features
        features = [
            data.get('Length', 1000.0) / 2000.0,
            data.get('Width', 300.0) / 1000.0,
            data.get('Height', 300.0) / 1000.0,
            data.get('GroundClearance', 50.0) / 100.0,
            data.get('SlantAngle', 25.0) / 90.0,
            data.get('FilletRadius', 0.0) / 200.0,
            data.get('Velocity', 40.0) / 100.0
        ]
        return np.array(features, dtype=np.float32)
    except Exception as e:
        print(f"Warning: Error parsing {filepath}: {e}")
        return np.zeros(7, dtype=np.float32)


def get_case_id(filename):
    match = re.search(r'\d+', filename)
    return match.group() if match else None


class AhmedProductionDataset(Dataset):
    def __init__(self, vtp_dir, info_dir, points=32000, mode="Train"):
        self.points = points
        self.file_list = glob.glob(os.path.join(vtp_dir, "*.vtp")) + \
                         glob.glob(os.path.join(vtp_dir, "*.vtu"))
        self.data_cache = []

        print(f"[{mode}] Indexing dataset...")

        for fpath in tqdm(self.file_list, desc=f"Loading {mode}"):
            try:
                case_id = get_case_id(os.path.basename(fpath))
                if not case_id: continue

                info_path = os.path.join(info_dir, f"case{case_id}_info.txt")
                if not os.path.exists(info_path): continue

                # 1. Load Physics (Global Token Data)
                phys_feats = parse_info_file(info_path)  # Shape: (7,)

                # 2. Load Geometry (Points)
                mesh = pv.read(fpath)
                pos = mesh.points.astype(np.float32)
                pos = pos - np.mean(pos, axis=0)  # Centering

                # 3. Load Targets
                targets = []
                if 'p' in mesh.point_data:
                    p = mesh.point_data['p']
                    if p.ndim == 1: p = p[:, None]
                    targets.append(p)
                if 'U' in mesh.point_data:
                    targets.append(mesh.point_data['U'])

                if not targets: continue
                target = np.concatenate(targets, axis=1).astype(np.float32)

                self.data_cache.append((pos, phys_feats, target))
            except Exception:
                continue

        print(f"[{mode}] Loaded {len(self.data_cache)} samples.")

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        pos, phys, target = self.data_cache[idx]

        # Sampling / Padding
        num_points = pos.shape[0]
        if num_points > self.points:
            indices = np.random.choice(num_points, self.points, replace=False)
            pos_out = pos[indices]
            tar_out = target[indices]
        else:
            pad_size = self.points - num_points
            if pad_size > 0:
                pos_out = np.concatenate([pos, pos[:pad_size]], axis=0)
                tar_out = np.concatenate([target, target[:pad_size]], axis=0)
            else:
                pos_out = pos
                tar_out = target

        # Return: Geometry (Nx3), Physics (7), Targets (Nx4)
        return torch.from_numpy(pos_out), torch.from_numpy(phys), torch.from_numpy(tar_out)