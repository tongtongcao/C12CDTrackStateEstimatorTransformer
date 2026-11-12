
import numpy as np
from typing import Final
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


# --------------------------------------------------
# CVT configuration
# --------------------------------------------------
STATE_DIM = 5

LAYER_IDS = [1, 2, 3, 4, 7, 9, 5, 6, 8]

# Token indices in the fixed sequence.
SVT_TOKENS = [0, 1, 2]
BMTC_TOKENS = [3, 4, 5]
BMTZ_TOKENS = [6, 7, 8]

STATE_NAMES = ["d0", "phi0", "kappa", "z0", "tandip"]


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class TrackDataset(Dataset):
    """
    Dataset for CVT track-state estimation.

    Each sample contains:

        features:   [9, 3]
        valid_mask: [9], True = valid detector layer
        state:      [5]

    Token representation:

        SVT:  (x, y, z)
        BMTC: (r, z, 0)       # Third coordinate is ignored
        BMTZ: (r, sin(phi), cos(phi))

    Feature normalization is performed independently
    for each token and feature.
    """

    def __init__(
        self,
        features,
        valid_mask,
        states,
        normalize=True,
        feature_stats=None,
        state_stats=None
    ):

        assert len(features) == len(valid_mask) == len(states)

        self.features = np.asarray(features, dtype=np.float32)
        self.valid_mask = np.asarray(valid_mask, dtype=bool)
        self.states = np.asarray(states, dtype=np.float32)

        if self.features.ndim != 3 or self.features.shape[1:] != (9, 3):
            raise ValueError(f"Expected features shape [N, 9, 3], got {self.features.shape}")

        if self.valid_mask.shape != (len(self.features), 9):
            raise ValueError("Invalid mask shape.")

        if self.states.shape != (len(self.features), 5):
            raise ValueError("Invalid track-state shape.")

        if not np.all(self.valid_mask.any(axis=1)):
            raise ValueError("Every track must have at least one valid token.")

        if not np.all(np.isfinite(self.features)):
            raise ValueError("Non-finite input features detected.")

        if not np.all(np.isfinite(self.states)):
            raise ValueError("Non-finite track states detected.")

        self.normalize = normalize

        # Feature statistics: [9, 3].
        self.feature_stats = feature_stats or {
            "mean": np.zeros((9, 3), dtype=np.float32),
            "std": np.ones((9, 3), dtype=np.float32)
        }

        self.feature_mean = np.asarray(self.feature_stats["mean"], dtype=np.float32)

        self.feature_std = np.asarray(self.feature_stats["std"], dtype=np.float32)

        if self.feature_mean.shape != (9, 3):
            raise ValueError("Feature mean must have shape [9, 3].")

        if self.feature_std.shape != (9, 3):
            raise ValueError("Feature std must have shape [9, 3].")

        if not np.all(np.isfinite(self.feature_mean)):
            raise ValueError("Invalid feature means.")

        if not np.all(np.isfinite(self.feature_std)):
            raise ValueError("Invalid feature standard deviations.")

        if np.any(self.feature_std <= 0):
            raise ValueError("Feature standard deviations must be positive.")

        # State statistics.
        self.state_stats = state_stats or {name: (0.0, 1.0) for name in STATE_NAMES}

        self.state_mean = np.asarray(
            [self.state_stats[k][0] for k in STATE_NAMES],
            dtype=np.float32
        )

        self.state_std = np.asarray(
            [self.state_stats[k][1] for k in STATE_NAMES],
            dtype=np.float32
        )

        if np.any(self.state_std <= 0):
            raise ValueError("State standard deviations must be positive.")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):

        features = self.features[idx].copy()
        valid_mask = self.valid_mask[idx].copy()
        state = self.states[idx].copy()

        if self.normalize:

            features = (features - self.feature_mean) / self.feature_std

            # Invalid tokens remain zero.
            features[~valid_mask] = 0.0

            state = (state - self.state_mean) / self.state_std

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(valid_mask, dtype=torch.bool),
            torch.tensor(state, dtype=torch.float32)
        )

    def denormalize_state(self, normed_state):

        if isinstance(normed_state, torch.Tensor):
            normed_state = normed_state.detach().cpu().numpy()

        return normed_state * self.state_std + self.state_mean


# --------------------------------------------------
# Collate function
# --------------------------------------------------

def collate_fn(batch):

    features, valid_masks, states = zip(*batch)

    return (
        torch.stack(features),
        torch.stack(valid_masks),
        torch.stack(states)
    )


# --------------------------------------------------
# Transformer model
# --------------------------------------------------

class TrackTransformer(pl.LightningModule):
    """
    Transformer-based CVT track-state estimator.

    Input:
        x:          [B, 9, 3]
        valid_mask: [B, 9], True = valid

    Output:
        [B, 5]

    State:
        (d0, phi0, kappa, z0, tandip)

    Detector-specific input embeddings:

        SVT:  3 -> hidden_dim
        BMTC: 2 -> hidden_dim
        BMTZ: 3 -> hidden_dim
    """
    NUM_TOKENS: Final[int] = 9

    def __init__(self, hidden_dim=32, nhead=4, num_layers=2, lr=1e-3, state_stats=None):

        super().__init__()

        self.save_hyperparameters(ignore=["state_stats"])

        self.lr = lr

        # --------------------------------------------------
        # Detector-specific feature embeddings
        # --------------------------------------------------

        self.svt_embedding = nn.Linear(3, hidden_dim)
        self.bmtc_embedding = nn.Linear(2, hidden_dim)
        self.bmtz_embedding = nn.Linear(3, hidden_dim)

        # --------------------------------------------------
        # Learnable layer embeddings
        # --------------------------------------------------

        self.layer_embedding = nn.Embedding(self.NUM_TOKENS, hidden_dim)

        # --------------------------------------------------
        # Transformer encoder
        # --------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4, batch_first=True)

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --------------------------------------------------
        # Track-state regression head
        # --------------------------------------------------

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, STATE_DIM)
        )

        # --------------------------------------------------
        # phi0 normalization for periodic loss
        # --------------------------------------------------

        if state_stats is None:
            phi_std = 1.0
        else:
            phi_std = float(state_stats["phi0"][1])

        if phi_std <= 0:
            raise ValueError("phi0 standard deviation must be positive.")

        self.register_buffer(
            "phi_std",
            torch.tensor(phi_std, dtype=torch.float32)
        )

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    def forward(self, x, valid_mask):
        """
        x:          [B, 9, 3]
        valid_mask: [B, 9], True = valid
        """

        # Detector-specific feature embedding.
        #
        # SVT:  [B, 3, 3] -> [B, 3, H]
        # BMTC: [B, 3, 2] -> [B, 3, H]
        # BMTZ: [B, 3, 3] -> [B, 3, H]

        svt = self.svt_embedding(x[:, 0:3, :])
        bmtc = self.bmtc_embedding(x[:, 3:6, 0:2])
        bmtz = self.bmtz_embedding(x[:, 6:9, :])

        # Reassemble tokens in the original fixed order.
        x_emb = torch.cat([svt, bmtc, bmtz], dim=1)

        # Add layer identity.
        layer_ids = torch.arange(self.NUM_TOKENS, device=x.device)
        layer_emb = self.layer_embedding(layer_ids)

        x_emb = x_emb + layer_emb.unsqueeze(0)

        # True in src_key_padding_mask means ignore.
        padding_mask = ~valid_mask

        # Transformer encoder.
        x_trans = self.transformer(x_emb, src_key_padding_mask=padding_mask)

        # Masked mean pooling.
        valid_weights = valid_mask.unsqueeze(-1).to(x_trans.dtype)
        x_trans = x_trans * valid_weights
        num_valid = valid_weights.sum(dim=1).clamp(min=1.0)
        x_pooled = x_trans.sum(dim=1) / num_valid

        # Regression.
        return self.fc(x_pooled)

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    def compute_loss(self, y_hat, y):
        diff = y_hat - y

        # Periodic phi0 residual.
        phi_diff = diff[:, 1] * self.phi_std
        phi_diff = torch.atan2(torch.sin(phi_diff), torch.cos(phi_diff))
        phi_diff = phi_diff / self.phi_std

        diff = torch.cat([diff[:, 0:1], phi_diff.unsqueeze(1), diff[:, 2:]], dim=1)

        return torch.mean(diff ** 2)

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    def training_step(self, batch, batch_idx):
        x, valid_mask, y = batch
        y_hat = self(x, valid_mask)
        loss = self.compute_loss(y_hat, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True,prog_bar=True)

        return loss

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    def validation_step(self, batch, batch_idx):
        x, valid_mask, y = batch
        y_hat = self(x, valid_mask)
        loss = self.compute_loss(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# --------------------------------------------------
# TorchScript inference wrapper
# --------------------------------------------------

class TrackTransformerWrapper(nn.Module):
    """
    Inference interface:

        features:   [B, 9, 3]
        valid_mask: [B, 9]

    Returns:
        predicted states: [B, 5]
    """

    def __init__(self, core_model):
        super().__init__()
        self.core = core_model

    def forward(self, x, valid_mask):
        return self.core(x, valid_mask)


# --------------------------------------------------
# Loss tracking callback
# --------------------------------------------------

class LossTracker(Callback):

    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        if "train_loss" in trainer.callback_metrics:
            self.train_losses.append(trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        if "val_loss" in trainer.callback_metrics:
            self.val_losses.append(trainer.callback_metrics["val_loss"].item())
