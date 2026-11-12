
import os
import time
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Subset

from trainer import *
from data import *
from plotter import Plotter


# --------------------------------------------------
# Configuration
# --------------------------------------------------

STATE_NAMES = ["d0", "phi0", "kappa", "z0", "tandip"]

NUM_TOKENS = 9

# Features used by each detector type.
TOKEN_FEATURES = {
    0: [0, 1, 2],  # SVT layer 1: x, y, z
    1: [0, 1, 2],  # SVT layer 2
    2: [0, 1, 2],  # SVT layer 3

    3: [0, 1],     # BMTC layer 4: r, z
    4: [0, 1],     # BMTC layer 7
    5: [0, 1],     # BMTC layer 9

    6: [0, 1, 2],  # BMTZ layer 5: r, sin(phi), cos(phi)
    7: [0, 1, 2],  # BMTZ layer 6
    8: [0, 1, 2],  # BMTZ layer 8
}


# --------------------------------------------------
# Command-line arguments
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="CVT Transformer Track-State Estimation")

    # Device selection
    parser.add_argument("--device", type=str, choices=["cpu", "gpu", "auto"], default="auto")

    # Input files
    parser.add_argument("inputs", type=str, nargs="*", default=["crossesTracks.csv"], help="One or more input CSV files")

    # Training parameters
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Transformer parameters
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)

    # Output
    parser.add_argument("--outdir", type=str, default="outputs/local")
    parser.add_argument("--end_name", type=str, default="")

    # Runtime options
    parser.add_argument("--no_train", action="store_true", help="Skip training and use a saved model")
    parser.add_argument("--enable_progress_bar", action="store_true")

    # Relative scale = ratio of scale factor of magnetic field between validation samples and training samples
    parser.add_argument("--scale", type=float, default=1.0)

    return parser.parse_args()


# --------------------------------------------------
# Random seed
# --------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)


# --------------------------------------------------
# Calculate feature normalization statistics
# --------------------------------------------------

def calculate_feature_stats(features, valid_mask):
    """
    Calculate normalization statistics independently
    for each detector layer and feature.

    Input:
        features:   [N, 9, 3]
        valid_mask: [N, 9]

    Output:
        mean: [9, 3]
        std:  [9, 3]

    Missing measurements are excluded.

    BMTC uses only r and z.
    Its third feature is assigned mean=0, std=1.
    """

    mean = np.zeros((NUM_TOKENS, 3), dtype=np.float32)
    std = np.ones((NUM_TOKENS, 3), dtype=np.float32)

    for token in range(NUM_TOKENS):

        # Select valid measurements for this layer.
        valid = valid_mask[:, token]

        if not np.any(valid):
            raise ValueError(f"No valid measurements found for token {token} (layer {LAYER_IDS[token]}).")

        for feature_idx in TOKEN_FEATURES[token]:
            values = features[valid, token, feature_idx]

            mean[token, feature_idx] = values.mean()
            std_value = values.std()

            # Avoid division by zero.
            std[token, feature_idx] = max(float(std_value), 1e-8)

    return {
        "mean": mean.tolist(),
        "std": std.tolist()
    }


# --------------------------------------------------
# Calculate track-state normalization statistics
# --------------------------------------------------

def calculate_state_stats(states):
    """
    Calculate statistics for:

        d0, phi0, kappa, z0, tandip

    phi0 uses a circular mean and wrapped residuals
    to account for its periodic nature.
    """

    state_stats = {}

    for i, name in enumerate(STATE_NAMES):
        values = states[:, i]

        mean = np.mean(values)
        std = np.std(values)
        std = max(float(std), 1e-8)

        state_stats[name] = (float(mean), std)

    return state_stats


# --------------------------------------------------
# Save normalization statistics
# --------------------------------------------------

def save_stats(feature_stats, state_stats, feature_stats_path, state_stats_path):
    with open(feature_stats_path, "w") as f:
        json.dump(feature_stats, f, indent=2)

    with open(state_stats_path, "w") as f:
        json.dump(state_stats, f, indent=2)


# --------------------------------------------------
# Load normalization statistics
# --------------------------------------------------

def load_stats(feature_stats_path, state_stats_path):
    with open(feature_stats_path, "r") as f:
        feature_stats = json.load(f)

    with open(state_stats_path, "r") as f:
        state_stats_raw = json.load(f)

    # JSON converts tuples to lists.
    state_stats = {
        name: tuple(state_stats_raw[name])
        for name in STATE_NAMES
    }

    return feature_stats, state_stats


# --------------------------------------------------
# Denormalize track states
# --------------------------------------------------
def denormalize_states(states, state_stats):
    result = states.copy()

    for i, name in enumerate(STATE_NAMES):
        mean, std = state_stats[name]
        result[:, i] = result[:, i] * std + mean

    # Wrap phi0 to [-pi, pi].
    result[:, 1] = np.arctan2(np.sin(result[:, 1]), np.cos(result[:, 1]))

    return result


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    set_seed(42)

    args = parse_args()

    inputs = args.inputs if args.inputs else ["sample.csv"]
    outDir = args.outdir
    os.makedirs(outDir, exist_ok=True)
    doTraining = not args.no_train
    end_name = args.end_name

    # Repository paths.
    BASE_DIR = Path(__file__).resolve().parent
    NETS_DIR = BASE_DIR / "nets"
    NETS_DIR.mkdir(parents=True, exist_ok=True)

    # Normalization files.
    feature_stats_out_path = (Path(outDir) / f"feature_stats.json")
    state_stats_out_path = (Path(outDir) / f"state_stats.json")

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    print("\nLoading CVT data...")

    startT_data = time.time()

    features_all = []
    masks_all = []
    states_all = []

    for fname in inputs:
        print(f"Loading {fname} ...")

        features, valid_mask, states = (read_tracks_with_crosses(fname))

        features_all.extend(features)
        masks_all.extend(valid_mask)
        states_all.extend(states)

    features_all = np.asarray(features_all, dtype=np.float32)

    masks_all = np.asarray(masks_all, dtype=bool)

    states_all = np.asarray(states_all, dtype=np.float32)

    if len(features_all) == 0:
        raise ValueError("No valid CVT tracks were loaded.")

    print("Number of tracks:", len(features_all))
    print("Features shape:", features_all.shape)
    print("Mask shape:", masks_all.shape)
    print("States shape:", states_all.shape)

    # --------------------------------------------------
    # Train / validation split
    # --------------------------------------------------
    n_samples = len(features_all)
    if n_samples < 2:
        raise ValueError("At least two tracks are required.")

    generator = torch.Generator().manual_seed(42)

    indices = torch.randperm(n_samples, generator=generator).numpy()

    train_size = int(0.8 * n_samples)
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:].tolist()
    if train_size == 0 or len(val_indices) == 0:
        raise ValueError("Invalid training/validation split.")

    print("\nTrain size:", len(train_indices))
    print("Validation size:", len(val_indices))

    # --------------------------------------------------
    # Normalization statistics
    # --------------------------------------------------

    if doTraining:
        print("\nCalculating normalization statistics...")

        # Use training samples only.
        train_features = features_all[train_indices]
        train_masks = masks_all[train_indices]
        train_states = states_all[train_indices]

        feature_stats = calculate_feature_stats(train_features, train_masks)
        state_stats = calculate_state_stats(train_states)

        save_stats(feature_stats, state_stats, feature_stats_out_path, state_stats_out_path)

        print("\nFeature statistics:")
        for token, layer in enumerate(LAYER_IDS):
            print(f"Layer {layer}: mean={feature_stats['mean'][token]}, std={feature_stats['std'][token]}")

        print("\nState statistics:")
        for name in STATE_NAMES:
            mean, std = state_stats[name]
            print(f"{name:>6s}: mean={mean:+.6g}, std={std:.6g}")

        print("\nSaved normalization statistics.")

    else:
        print("\nLoading normalization statistics from nets/...")

        feature_stats_path = (NETS_DIR / f"feature_stats.json")
        state_stats_path = (NETS_DIR / f"state_stats_cvt.json")

        for path in (feature_stats_path, state_stats_path):
            if not path.exists():
                raise FileNotFoundError(f"Normalization statistics not found: {path}")

        feature_stats, state_stats = load_stats(feature_stats_path, state_stats_path)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------
    dataset = TrackDataset(features=features_all, valid_mask=masks_all, states=states_all, normalize=True, feature_stats=feature_stats, state_stats=state_stats)

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    batch_size = args.batch_size

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Inspect one batch.
    features_sample, mask_sample, states_sample = next(iter(train_loader))

    print("\nBatch inspection:")
    print("Features:", features_sample.shape)
    print("Valid mask:", mask_sample.shape)
    print("States:", states_sample.shape)

    endT_data = time.time()

    print(f"\nData preparation took {endT_data - startT_data:.2f} seconds.")

    # --------------------------------------------------
    # Plotting
    # --------------------------------------------------
    plotter = Plotter(print_dir=outDir, end_name=end_name)

    # --------------------------------------------------
    # Model initialization
    # --------------------------------------------------

    if args.hidden_dim % args.nhead != 0:
        raise ValueError("hidden_dim must be divisible by nhead.")

    model = TrackTransformer(hidden_dim=args.hidden_dim, nhead=args.nhead, num_layers=args.num_layers, lr=args.lr, state_stats=state_stats)

    loss_tracker = LossTracker()

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    if doTraining:
        if args.device == "cpu":
            accelerator, devices = "cpu", 1
        elif args.device == "gpu":
            if torch.cuda.is_available():
                accelerator, devices = "gpu", 1
            else:
                print("GPU unavailable. Using CPU.")
                accelerator, devices = "cpu", 1
        else:
            if torch.cuda.is_available():
                accelerator, devices = "gpu", 1
            else:
                accelerator, devices = "cpu", 1

        print(f"\nUsing accelerator={accelerator}, devices={devices}")

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            strategy="auto",
            max_epochs=args.max_epochs,
            enable_progress_bar=args.enable_progress_bar,
            log_every_n_steps=100,
            enable_checkpointing=False,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=0,
            logger=False,
            callbacks=[loss_tracker]
        )

        print("\nTraining CVT Transformer...")

        startT_train = time.time()

        trainer.fit(model, train_loader, val_loader)

        endT_train = time.time()

        print(f"\nTraining took {(endT_train - startT_train) / 60:.2f} minutes.")

        # Plot training loss.
        plotter.plotTrainLoss(loss_tracker)

        # --------------------------------------------------
        # Export TorchScript
        # --------------------------------------------------

        print("\nExporting TorchScript model...")

        model = model.to("cpu")
        model.eval()

        wrapper_model = TrackTransformerWrapper(model)
        wrapper_model.eval()

        model_file = (Path(outDir) / f"transformer_cvt_{end_name}.pt")

        torchscript_model = torch.jit.script(wrapper_model)
        torchscript_model.save(model_file)

        print("Saved model:", model_file)

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    else:
        model_file = (NETS_DIR / f"transformer_default.pt")

    model_file = model_file.resolve()

    print("\nLoading model from:", model_file)

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    model = torch.jit.load(str(model_file))
    model.eval()

    # --------------------------------------------------
    # Validation inference
    # --------------------------------------------------
    val_loader_test = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    all_preds = []
    all_targets = []

    print("\nRunning validation inference...")

    startT_test = time.time()

    with torch.no_grad():
        for features_batch, mask_batch, state_batch in val_loader_test:
            features_batch = features_batch.to(device)
            mask_batch = mask_batch.to(device)
            state_batch = state_batch.to(device)
            y_pred = model(features_batch, mask_batch)
            all_preds.append(y_pred.cpu())
            all_targets.append(state_batch.cpu())

    endT_test = time.time()

    print(f"Validation inference took {endT_test - startT_test:.2f} seconds.")

    # --------------------------------------------------
    # Collect predictions
    # --------------------------------------------------
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # --------------------------------------------------
    # Denormalize
    # --------------------------------------------------
    all_preds_denorm = denormalize_states(all_preds, state_stats)
    all_targets_denorm = denormalize_states(all_targets, state_stats)

    # --------------------------------------------------
    # Diagnostic plots
    # --------------------------------------------------

    # NOTE:
    # The original DC plotter uses DC-specific parameter
    # names and plotting ranges. Update plotter.py for
    # CVT before enabling these two methods.
    #
    plotter.plot_residuals(all_targets_denorm, all_preds_denorm, args.scale)
    plotter.plot_pred_target(all_targets_denorm, all_preds_denorm, args.scale)

    print("\nValidation completed.")
    print("Prediction shape:", all_preds_denorm.shape)
    print("Target shape:", all_targets_denorm.shape)
    print("\nMean absolute residuals:")

    residuals = all_preds_denorm - all_targets_denorm
    residuals[:, 1] = np.arctan2(np.sin(residuals[:, 1]), np.cos(residuals[:, 1]))

    for i, name in enumerate(STATE_NAMES):
        print(f"{name:>6s}: {np.mean(np.abs(residuals[:, i])):.6g}")


# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    main()
