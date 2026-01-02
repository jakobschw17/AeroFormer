import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import re

# Assuming these are in your local directory structure
from dataset import AhmedProductionDataset
from model import ProductionTransformer
from utils import save_slice_visuals

# ==========================================
# CONFIGURATION
# ==========================================
torch.set_float32_matmul_precision('high')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    'batch_size': 4,
    'points_per_sample': 32000,
    'model_dim': 128,
    'num_layers': 6,
    'num_heads': 4,
    'learning_rate': 5e-5,
    'epochs': 850,
    'vis_interval': 10,
    'vis_dir': 'visuals_production',
    'checkpoint_dir': 'checkpoints',
    'resume_checkpoint': "model_ep700.pth" #in folder checkpoints
}

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(CURRENT_DIR, "physicsnemo_ahmed_body_dataset_vv1")

DIRS = {
    'train_vtp': os.path.join(BASE_PATH, "dataset", "train"),
    'train_info': os.path.join(BASE_PATH, "dataset", "train_info"),
    'test_vtp': os.path.join(BASE_PATH, "dataset", "test"),
    'test_info': os.path.join(BASE_PATH, "dataset", "test_info")
}


def main():
    print(f"Starting Training on {device}")
    os.makedirs(CONFIG['vis_dir'], exist_ok=True)
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

    # 1. Dataset
    # ------------------------------------------
    train_ds = AhmedProductionDataset(DIRS['train_vtp'], DIRS['train_info'],
                                      points=CONFIG['points_per_sample'], mode="Train")
    test_ds = AhmedProductionDataset(DIRS['test_vtp'], DIRS['test_info'],
                                     points=CONFIG['points_per_sample'], mode="Test")

    if len(train_ds) == 0:
        print("Error: No training data found.")
        return

    # Check dimensions
    dummy_geo, dummy_phys, dummy_y = train_ds[0]
    geo_dim = dummy_geo.shape[1]  # 3 (x,y,z)
    phys_dim = dummy_phys.shape[0]  # 7 (physics)
    output_dim = dummy_y.shape[1]  # 4 (p, ux, uy, uz)

    print(f"Dims: Geo {geo_dim} + Phys {phys_dim} -> Out {output_dim}")

    # 2. Model
    # ------------------------------------------
    model = ProductionTransformer(
        geo_dim, phys_dim, output_dim,
        model_dim=CONFIG['model_dim'],
        num_layers=CONFIG['num_layers'],
        num_heads=CONFIG['num_heads']
    ).to(device)


    start_epoch = 0

    if CONFIG['resume_checkpoint']:
        resume_path = os.path.join(CONFIG['checkpoint_dir'], CONFIG['resume_checkpoint'])

        if os.path.exists(resume_path):
            print(f"Resuming from {resume_path}...")
            model.load_state_dict(torch.load(resume_path, map_location=device))
            try:
                start_epoch = int(re.search(r'\d+', CONFIG['resume_checkpoint']).group())
                print(f"Successfully resumed. Next epoch will be {start_epoch + 1}.")
            except:
                print("Warning: Could not parse epoch number. Starting counter at 0.")
        else:
            print(f"Warning: Checkpoint '{resume_path}' not found. Starting from scratch.")
    else:
        print("No checkpoint specified. Starting from scratch.")

    # 4. Training Setup
    # ------------------------------------------
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                              shuffle=True, num_workers=0, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-5)
    criterion = nn.MSELoss()

    print(f"Training started...")

    # 5. Training Loop
    # ------------------------------------------
    for epoch in range(start_epoch + 1, CONFIG['epochs'] + 1):
        model.train()
        loop = tqdm(train_loader, desc=f"Ep {epoch}", leave=True)
        epoch_loss = 0

        # Unpack: Geo, Phys, Target
        for geo, phys, target in loop:
            geo, phys, target = geo.to(device), phys.to(device), target.to(device)

            optimizer.zero_grad()

            # Pass both inputs
            pred = model(geo, phys)

            loss = criterion(pred, target)

            if torch.isnan(loss):
                print("NaN Loss detected! Stopping training.")
                return

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix({'loss': f"{loss.item():.0f}"})

        # Visualization and Saving
        if epoch % CONFIG['vis_interval'] == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch} | Avg Loss: {avg_loss:.0f}")

            # Save visuals
            save_slice_visuals(model, test_ds, epoch, criterion, device, CONFIG['vis_dir'])

            # Save checkpoint
            save_path = os.path.join(CONFIG['checkpoint_dir'], f"model_ep{epoch}.pth")
            torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    main()