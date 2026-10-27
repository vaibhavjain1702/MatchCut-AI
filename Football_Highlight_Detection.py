# %% [markdown]
# # 🏟️ AI-Based Football Highlight Detection System
# ## Deep Learning Pipeline: CNN Spatial Features + Temporal Modeling (Bi-LSTM / Transformers)
#
# **Academic Context & Architecture Explanation**
# This project fulfills Advanced Deep Learning requirements by implementing a two-stage computer vision and sequential modeling pipeline:
# 
# 1. **Spatial Feature Extraction (CNN)**: Uses a pre-trained ResNet-152 convolutional neural network to extract 2048-dimensional spatial feature vectors from raw video frames at 2 FPS, then applies PCA to reduce to 512 dimensions (matching SoccerNet's feature format). This isolates the spatial semantics (e.g., players, ball, field lines) from the raw pixel data.
# 2. **Temporal Modeling (RNN / Attention)**: Processes the extracted frame features using sliding windows (sequences of 15 seconds). It compares three temporal architectures:
#    - **Baseline CNN**: Processes independently via dense layers.
#    - **Bi-Directional LSTM**: Captures forward and backward temporal dependencies across frames.
#    - **Transformer Encoder**: Uses multi-head self-attention to dynamically weigh the importance of different frames within the window, achieving state-of-the-art context awareness.
# 3. **Optimization & Highlight Generation**: Uses PyTorch with AdamW optimizer, learning rate scheduling (ReduceLROnPlateau), and Early Stopping. Outputs are passed through a Knapsack DP algorithm to extract a strictly time-bounded highlight reel.
#
# **Dataset**: SoccerNet v2 (Action Spotting subset)
# **Output**: End-to-End MP4 generation from raw video input
# ---
# %% [markdown]
# ## Cell 1: Setup & Mount Google Drive

# %%
# ============================================================
# CELL 1: SETUP & MOUNT GOOGLE DRIVE
# ============================================================
import os
import sys
import json
import glob
import random
import shutil
import datetime
import warnings
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)

warnings.filterwarnings('ignore')

# --- Mount Google Drive (Colab only) ---
try:
    from google.colab import drive
    ON_COLAB = True
    try:
        drive.mount('/content/drive')
        PROJECT_DIR = '/content/drive/MyDrive/DL_Project'
        print("✅ Running on Google Colab (Drive Mounted)")
    except Exception as e:
        print(f"⚠️ Drive mount failed: {e}")
        print("⚠️ Falling back to Colab local storage (will be wiped after session)")
        PROJECT_DIR = '/content/DL_Project'
except ImportError:
    ON_COLAB = False
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    print("✅ Running locally")

# Create directories
for d in ['data', 'checkpoints', 'results', 'outputs']:
    os.makedirs(os.path.join(PROJECT_DIR, d), exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# %% [markdown]
# ## Cell 2: Install Dependencies & Download Data

# %%
# ============================================================
# CELL 2: INSTALL DEPENDENCIES (Colab only)
# ============================================================
if ON_COLAB:
    os.system('pip install SoccerNet moviepy -q')
    print("✅ Dependencies installed")

# %% [markdown]
# ## Cell 2b: Backup & Restore Utilities

# %%
# ============================================================
# CELL 2b: BACKUP & RESTORE UTILITIES
# ============================================================
import zipfile

def backup_and_download(label="checkpoint"):
    """
    Zip all checkpoints + results and auto-download to your machine.
    Call this after EVERY training cell to be 100% safe.

    Args:
        label: A short tag to identify this backup (e.g., 'after_baseline')
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    zip_name = f'backup_{label}_{timestamp}.zip'
    zip_path = f'/content/{zip_name}'

    print(f"\n📦 Creating backup: {zip_name}...")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Backup ALL checkpoints (best + latest + test_results)
        ckpt_dir = CONFIG['checkpoint_dir']
        if os.path.exists(ckpt_dir):
            for f in os.listdir(ckpt_dir):
                fpath = os.path.join(ckpt_dir, f)
                if os.path.isfile(fpath):
                    zipf.write(fpath, f'checkpoints/{f}')
                    print(f"   ✅ Packed: checkpoints/{f}")

        # Backup results (plots, JSON)
        res_dir = CONFIG['results_dir']
        if os.path.exists(res_dir):
            for f in os.listdir(res_dir):
                fpath = os.path.join(res_dir, f)
                if os.path.isfile(fpath):
                    zipf.write(fpath, f'results/{f}')

        # Backup outputs
        out_dir = CONFIG['outputs_dir']
        if os.path.exists(out_dir):
            for f in os.listdir(out_dir):
                fpath = os.path.join(out_dir, f)
                if os.path.isfile(fpath):
                    zipf.write(fpath, f'outputs/{f}')

    zip_size = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"   📦 Backup size: {zip_size:.1f} MB")

    # Auto-download to your Mac
    if ON_COLAB:
        from google.colab import files
        print("   ⬇️  Downloading to your Mac... check your Downloads folder!")
        files.download(zip_path)

    print(f"   ✅ Backup complete: {zip_name}")
    return zip_path

def restore_from_backup(zip_path_on_colab='/content/uploaded_backup.zip'):
    """
    Restore checkpoints from a previously downloaded backup zip.

    Usage:
        1. Upload your backup zip to Colab (use the file upload button in the sidebar)
        2. Call: restore_from_backup('/content/backup_after_bilstm_20260417_1700.zip')
        3. Re-run Cells 1-8, then 8a/8b/8c — they will auto-skip already-trained models!
    """
    if not os.path.exists(zip_path_on_colab):
        print(f"❌ File not found: {zip_path_on_colab}")
        print("   Upload the zip file first using the Colab file browser (left sidebar).")
        return False

    print(f"🔄 Restoring from: {zip_path_on_colab}...")

    with zipfile.ZipFile(zip_path_on_colab, 'r') as zipf:
        # Extract checkpoints to the project checkpoint dir
        for member in zipf.namelist():
            if member.startswith('checkpoints/'):
                filename = os.path.basename(member)
                if filename:
                    target = os.path.join(CONFIG['checkpoint_dir'], filename)
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with zipf.open(member) as src, open(target, 'wb') as dst:
                        dst.write(src.read())
                    print(f"   ✅ Restored: {filename}")

            elif member.startswith('results/'):
                filename = os.path.basename(member)
                if filename:
                    target = os.path.join(CONFIG['results_dir'], filename)
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with zipf.open(member) as src, open(target, 'wb') as dst:
                        dst.write(src.read())

            elif member.startswith('outputs/'):
                filename = os.path.basename(member)
                if filename:
                    target = os.path.join(CONFIG['outputs_dir'], filename)
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with zipf.open(member) as src, open(target, 'wb') as dst:
                        dst.write(src.read())

    print("   ✅ Restore complete! Now re-run Cells 8a/8b/8c — trained models will auto-skip.")
    return True

print("✅ Backup & Restore utilities defined")
print("   Use: backup_and_download('after_baseline') — to save and download checkpoints")
print("   Use: restore_from_backup('/content/your_backup.zip') — to restore checkpoints")

# %% [markdown]
# ## Cell 3: Configuration

# %%
# ============================================================
# CELL 3: CONFIGURATION — ALL HYPERPARAMETERS IN ONE PLACE
# ============================================================
CONFIG = {
    # --- Paths ---
    'project_dir': PROJECT_DIR,
    'data_dir': os.path.join(PROJECT_DIR, 'data', 'SoccerNet_subset'),
    'checkpoint_dir': os.path.join(PROJECT_DIR, 'checkpoints'),
    'results_dir': os.path.join(PROJECT_DIR, 'results'),
    'outputs_dir': os.path.join(PROJECT_DIR, 'outputs'),

    # --- Data ---
    'feature_dim': 512,           # ResNet feature dimension from SoccerNet
    'target_fps': 2,              # SoccerNet features are at 2 FPS
    'seq_len': 30,                # Sliding window = 15 seconds @ 2fps
    'seq_stride': 15,             # Stride for sliding window (50% overlap)
    'max_matches_train': 60,      # Using 60/20/20 because we are only downloading the smaller 100-match 'test' split
    'max_matches_val': 20,        # to completely guarantee it NEVER hits the 112GB Colab disk limit
    'max_matches_test': 20,       

    # --- Event Classes ---
    'event_classes': ['Goal', 'Cards', 'Substitution', 'Background'],
    'num_classes': 4,
    # Map SoccerNet labels to our simplified classes
    'label_map': {
        'Goal': 'Goal',
        'Penalty': 'Goal',
        'Own goal - Loss of Ball Rec': 'Goal',
        'Yellow card': 'Cards',
        'Red card': 'Cards',
        'Yellow card -> Red card': 'Cards',
        'Substitution': 'Substitution',
        # Everything else becomes Background
    },
    'event_window': 10,           # ±frames around event to label (±5 seconds)

    # --- Model ---
    'hidden_dim': 256,            # LSTM hidden size
    'num_heads': 8,               # Transformer attention heads
    'num_layers': 2,              # LSTM/Transformer layers
    'ff_dim': 1024,               # Transformer feedforward dim
    'dropout': 0.4,               # Dropout rate

    # --- Training ---
    # ⚡ FAST DEBUG MODE: Set to True for quick testing, False for real training
    'fast_debug_mode': False,

    'lr': 5e-4,                   # Learning rate
    'weight_decay': 1e-5,         # L2 regularization
    'batch_size': 32,             # Batch size
    'epochs': 80,                 # Maximum epochs (early stopping usually kicks in earlier)
    'patience': 15,               # Early stopping patience
    'clip_grad': 5.0,             # Gradient clipping
    'reg_factor': 0.15,           # Length regularization target
    'reg_lambda': 0.5,            # Length regularization weight

    # --- Device ---
    'device': DEVICE,
    'seed': 42,
}

# Override subset settings if we are in debug mode
if CONFIG['fast_debug_mode']:
    print("⚠️ FAST DEBUG MODE IS ON: Using extremely small configurations for rapid testing!")
    CONFIG['max_matches_train'] = 5
    CONFIG['max_matches_val'] = 2
    CONFIG['max_matches_test'] = 2
    CONFIG['epochs'] = 2

# Set seeds for reproducibility
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
random.seed(CONFIG['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG['seed'])

print("✅ Configuration loaded")
print(f"   Classes: {CONFIG['event_classes']}")
print(f"   Sequence length: {CONFIG['seq_len']} frames ({CONFIG['seq_len']/CONFIG['target_fps']:.0f} seconds)")
print(f"   Device: {CONFIG['device']}")

# %% [markdown]
# ## Cell 4: Download SoccerNet Data (Run ONCE)

# %%
# ============================================================
# CELL 4: DOWNLOAD SOCCERNET DATA (RUN THIS ONLY ONCE)
# ============================================================

def download_soccernet_subset():
    """Download SoccerNet features and labels, then copy a subset to Drive."""
    from SoccerNet.Downloader import SoccerNetDownloader

    local_dir = "/content/SoccerNet_temp" if ON_COLAB else os.path.join(PROJECT_DIR, 'data', 'SoccerNet_temp')
    target_dir = CONFIG['data_dir']

    if os.path.exists(target_dir) and len(glob.glob(os.path.join(target_dir, '**', 'Labels-v2.json'), recursive=True)) > 10:
        count = len(glob.glob(os.path.join(target_dir, '**', 'Labels-v2.json'), recursive=True))
        print(f"✅ SoccerNet subset already exists with {count} matches. Skipping download.")
        return

    print("📥 Downloading SoccerNet features (this may take 30-60 minutes)...")

    downloader = SoccerNetDownloader(LocalDirectory=local_dir)

    # Force download ONLY our exact target compressed features and labels
    downloader.downloadGames(
        files=["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy", "Labels-v2.json"],
        split=["test"]
    )

    # Find all match directories that have both features AND labels
    all_matches = []
    for root, dirs, files in os.walk(local_dir):
        has_features = '1_ResNET_TF2_PCA512.npy' in files or '2_ResNET_TF2_PCA512.npy' in files
        has_labels = 'Labels-v2.json' in files
        if has_features and has_labels:
            all_matches.append(root)

    print(f"📊 Found {len(all_matches)} complete matches")

    # Select subset
    random.seed(CONFIG['seed'])
    random.shuffle(all_matches)
    total_needed = CONFIG['max_matches_train'] + CONFIG['max_matches_val'] + CONFIG['max_matches_test']
    selected = all_matches[:min(total_needed, len(all_matches))]

    # Copy to Drive
    print(f"📦 Copying {len(selected)} matches to Google Drive...")
    for i, match_dir in enumerate(selected):
        rel_path = os.path.relpath(match_dir, local_dir)
        dest = os.path.join(target_dir, rel_path)
        os.makedirs(dest, exist_ok=True)

        for f in ['1_ResNET_TF2_PCA512.npy', '2_ResNET_TF2_PCA512.npy', 'Labels-v2.json']:
            src = os.path.join(match_dir, f)
            if os.path.exists(src):
                shutil.copy2(src, dest)

        if (i + 1) % 20 == 0:
            print(f"   Copied {i + 1}/{len(selected)} matches...")

    # Cleanup temp directory on Colab local disk
    if ON_COLAB and os.path.exists(local_dir):
        shutil.rmtree(local_dir)
        print("🧹 Cleaned up temporary download")

    total_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, dn, fn in os.walk(target_dir)
        for f in fn
    ) / (1024 ** 3)

    print(f"✅ SoccerNet subset ready: {len(selected)} matches, {total_size:.2f} GB on Drive")

# Uncomment the line below to download (run ONCE, then comment out)
# download_soccernet_subset()

# %% [markdown]
# ## Cell 5: Dataset Class

# %%
# ============================================================
# CELL 5: SOCCERNET DATASET CLASS
# ============================================================

class SoccerNetHighlightDataset(Dataset):
    """
    Loads SoccerNet pre-extracted ResNet features and event labels.
    Creates sliding window sequences with frame-level event labels.
    """

    def __init__(self, data_dir, split='train', config=None):
        """
        Args:
            data_dir: Path to SoccerNet_subset directory
            split: 'train', 'val', or 'test'
            config: Configuration dictionary
        """
        self.config = config or CONFIG
        self.split = split
        self.sequences = []
        self.labels = []
        self.match_info = []

        # Find all match directories
        all_matches = self._find_matches(data_dir)

        if len(all_matches) == 0:
            print(f"⚠️  No matches found in {data_dir}. Using synthetic data for testing.")
            self._create_synthetic_data()
            return

        # Split matches into train/val/test
        random.seed(self.config['seed'])
        shuffled = all_matches.copy()
        random.shuffle(shuffled)

        n_train = self.config['max_matches_train']
        n_val = self.config['max_matches_val']

        if split == 'train':
            matches = shuffled[:n_train]
        elif split == 'val':
            matches = shuffled[n_train:n_train + n_val]
        else:  # test
            matches = shuffled[n_train + n_val:]

        print(f"📂 Loading {split} set: {len(matches)} matches...")

        for i, match_dir in enumerate(matches):
            self._process_match(match_dir)
            if (i + 1) % 20 == 0:
                print(f"   Processed {i + 1}/{len(matches)} matches...")

        print(f"✅ {split} set: {len(self.sequences)} sequences from {len(matches)} matches")

        # Compute class weights for imbalanced data
        if split == 'train':
            self._compute_class_weights()

    def _find_matches(self, data_dir):
        """Find all match directories with features and labels."""
        matches = []
        if not os.path.exists(data_dir):
            return matches
        for root, dirs, files in os.walk(data_dir):
            if 'Labels-v2.json' in files and (
                '1_ResNET_TF2_PCA512.npy' in files or '2_ResNET_TF2_PCA512.npy' in files
            ):
                matches.append(root)
        return sorted(matches)

    def _process_match(self, match_dir):
        """Load features and labels for a single match, create sequences."""
        # Load labels
        labels_path = os.path.join(match_dir, 'Labels-v2.json')
        with open(labels_path, 'r') as f:
            labels_data = json.load(f)

        # Process each half
        for half in [1, 2]:
            feat_path = os.path.join(match_dir, f'{half}_ResNET_TF2_PCA512.npy')
            if not os.path.exists(feat_path):
                continue

            features = np.load(feat_path)  # Shape: (num_frames, 512)
            num_frames = features.shape[0]

            # Create frame-level labels (default = Background = last class)
            frame_labels = np.full(num_frames, self.config['num_classes'] - 1, dtype=np.int64)

            # Parse event annotations for this half
            for ann in labels_data.get('annotations', []):
                game_time = ann.get('gameTime', '')
                label = ann.get('label', '')

                # Parse half number and time
                try:
                    half_str, time_str = game_time.split(' - ')
                    ann_half = int(half_str.strip())
                    if ann_half != half:
                        continue

                    minutes, seconds = time_str.strip().split(':')
                    total_seconds = int(minutes) * 60 + int(seconds)
                    frame_idx = int(total_seconds * self.config['target_fps'])
                except (ValueError, IndexError):
                    continue

                # Map label to our simplified classes
                mapped_label = self.config['label_map'].get(label, None)
                if mapped_label is None:
                    continue  # Skip unmapped events

                class_idx = self.config['event_classes'].index(mapped_label)

                # Label a window of frames around the event
                window = self.config['event_window']
                start = max(0, frame_idx - window)
                end = min(num_frames, frame_idx + window + 1)
                frame_labels[start:end] = class_idx

            # Create sliding window sequences
            seq_len = self.config['seq_len']
            stride = self.config['seq_stride']

            for start_idx in range(0, num_frames - seq_len, stride):
                end_idx = start_idx + seq_len
                seq_features = features[start_idx:end_idx]
                seq_labels = frame_labels[start_idx:end_idx]

                # Use the CENTER frame's label as the sequence label
                center_label = seq_labels[seq_len // 2]

                self.sequences.append(torch.tensor(seq_features, dtype=torch.float32))
                self.labels.append(center_label)
                self.match_info.append({
                    'match_dir': match_dir,
                    'half': half,
                    'start_frame': start_idx,
                    'end_frame': end_idx,
                })

    def _create_synthetic_data(self):
        """Create synthetic data for testing when SoccerNet is not available."""
        print("🧪 Creating synthetic dataset for pipeline testing...")
        num_samples = 2000 if self.split == 'train' else 500
        seq_len = self.config['seq_len']
        feat_dim = self.config['feature_dim']

        for i in range(num_samples):
            features = torch.randn(seq_len, feat_dim)
            # Create imbalanced labels (90% background, 10% events)
            if random.random() < 0.9:
                label = self.config['num_classes'] - 1  # Background
            else:
                label = random.randint(0, self.config['num_classes'] - 2)

            self.sequences.append(features)
            self.labels.append(label)

        self._compute_class_weights()

    def _compute_class_weights(self):
        """Compute inverse-frequency class weights for imbalanced training."""
        label_counts = np.bincount(self.labels, minlength=self.config['num_classes'])
        total = len(self.labels)

        # Inverse frequency weighting
        weights = np.zeros(self.config['num_classes'])
        for i in range(self.config['num_classes']):
            if label_counts[i] > 0:
                weights[i] = total / (self.config['num_classes'] * label_counts[i])
            else:
                weights[i] = 1.0

        self.class_weights = torch.tensor(weights, dtype=torch.float32)

        print(f"   Class distribution:")
        for i, cls in enumerate(self.config['event_classes']):
            pct = label_counts[i] / total * 100 if total > 0 else 0
            print(f"     {cls}: {label_counts[i]} ({pct:.1f}%) — weight: {weights[i]:.2f}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def create_dataloaders(config):
    """Create train, val, test dataloaders."""
    data_dir = config['data_dir']

    train_dataset = SoccerNetHighlightDataset(data_dir, split='train', config=config)
    val_dataset = SoccerNetHighlightDataset(data_dir, split='val', config=config)
    test_dataset = SoccerNetHighlightDataset(data_dir, split='test', config=config)

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_dataset

print("✅ Dataset class defined")

# %% [markdown]
# ## Cell 6: Model Definitions

# %%
# ============================================================
# CELL 6: MODEL DEFINITIONS — Baseline, Bi-LSTM, Transformer
# ============================================================

# -------------------------------------------------------
# MODEL A: CNN-Only Baseline (no temporal modeling)
# -------------------------------------------------------
class BaselineCNN(nn.Module):
    """
    Baseline: Classifies each frame independently using only spatial features.
    No temporal context — serves as ablation baseline.
    """
    def __init__(self, config):
        super().__init__()
        self.name = "CNN_Baseline"
        feat_dim = config['feature_dim']
        num_classes = config['num_classes']
        dropout = config['dropout']

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2.0))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, feature_dim)
        We only use the CENTER frame for baseline (no temporal context).
        """
        center = x[:, x.shape[1] // 2, :]  # (batch_size, feature_dim)
        return self.classifier(center)      # (batch_size, num_classes)

# -------------------------------------------------------
# MODEL B: CNN + Bi-LSTM (Primary Model)
# -------------------------------------------------------
class BiLSTMClassifier(nn.Module):
    """
    CNN features → Bidirectional LSTM → Event Classification.
    Captures temporal dynamics in both forward and backward directions.
    """
    def __init__(self, config):
        super().__init__()
        self.name = "CNN_BiLSTM"
        feat_dim = config['feature_dim']
        hidden_dim = config['hidden_dim']
        num_layers = config['num_layers']
        num_classes = config['num_classes']
        dropout = config['dropout']

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        lstm_output_dim = hidden_dim * 2  # bidirectional doubles the output

        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        # LSTM weight initialization
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
                # Set forget gate bias to 1 (helps LSTM remember)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)

        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2.0))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, feature_dim)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the center frame's output (has context from both directions)
        center_idx = x.shape[1] // 2
        center_output = lstm_out[:, center_idx, :]  # (batch_size, hidden_dim*2)

        return self.classifier(center_output)       # (batch_size, num_classes)

# -------------------------------------------------------
# MODEL C: CNN + Transformer Encoder (Advanced Model)
# -------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x shape: (batch_size, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerClassifier(nn.Module):
    """
    CNN features → Positional Encoding → Transformer Encoder → Event Classification.
    Uses self-attention to capture long-range dependencies between frames.
    """
    def __init__(self, config):
        super().__init__()
        self.name = "CNN_Transformer"
        feat_dim = config['feature_dim']
        num_heads = config['num_heads']
        num_layers = config['num_layers']
        ff_dim = config['ff_dim']
        num_classes = config['num_classes']
        dropout = config['dropout']

        self.pos_encoder = PositionalEncoding(feat_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2.0))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, feature_dim)
        """
        x = self.pos_encoder(x)
        transformer_out = self.transformer_encoder(x)  # (batch_size, seq_len, feat_dim)

        # Use center frame output
        center_idx = x.shape[1] // 2
        center_output = transformer_out[:, center_idx, :]  # (batch_size, feat_dim)

        return self.classifier(center_output)              # (batch_size, num_classes)

def get_model(model_name, config):
    """Factory function to create models."""
    models = {
        'baseline': BaselineCNN,
        'bilstm': BiLSTMClassifier,
        'transformer': TransformerClassifier,
    }
    model = models[model_name](config).to(config['device'])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📐 {model.name}: {num_params:,} trainable parameters")
    return model

print("✅ All 3 models defined: BaselineCNN, BiLSTMClassifier, TransformerClassifier")

# %% [markdown]
# ## Cell 7: Training Engine

# %%
# ============================================================
# CELL 7: TRAINING ENGINE — Train, Evaluate, Checkpoint
# ============================================================

class TrainingEngine:
    """Handles training, evaluation, checkpointing, and resumption."""

    def __init__(self, model, config, class_weights=None):
        self.model = model
        self.config = config
        self.device = config['device']
        self.model_name = model.name

        # Loss with class weights to handle imbalance
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )

        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'val_map': [], 'lr': []
        }

        self.best_val_map = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.start_epoch = 0

    def length_regularization(self, logits):
        """Prevent model from marking everything as an event."""
        probs = torch.softmax(logits, dim=1)
        # Probability assigned to non-background classes
        event_probs = 1.0 - probs[:, -1]  # Background is last class
        return torch.abs(event_probs.mean() - self.config['reg_factor'])

    def train_one_epoch(self, train_loader):
        """Train for one epoch, return average loss and accuracy."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device).long()

            self.optimizer.zero_grad()

            logits = self.model(features)
            ce_loss = self.criterion(logits, labels)
            reg_loss = self.length_regularization(logits)
            loss = ce_loss + self.config['reg_lambda'] * reg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config['clip_grad']
            )
            self.optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total if total > 0 else 0
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, val_loader):
        """Evaluate model, return loss, accuracy, and per-class mAP."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        for features, labels in val_loader:
            features = features.to(self.device)
            labels = labels.to(self.device).long()

            logits = self.model(features)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        avg_loss = total_loss / max(len(val_loader), 1)
        accuracy = correct / total if total > 0 else 0

        # Compute per-class Average Precision
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        aps = []
        for c in range(self.config['num_classes'] - 1):  # Exclude background
            binary_labels = (all_labels == c).astype(int)
            if binary_labels.sum() > 0:
                ap = average_precision_score(binary_labels, all_probs[:, c])
                aps.append(ap)

        mean_ap = np.mean(aps) if aps else 0.0

        return avg_loss, accuracy, mean_ap, np.array(all_preds), np.array(all_labels), np.array(all_probs)

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint to Drive."""
        ckpt_dir = self.config['checkpoint_dir']
        os.makedirs(ckpt_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_name': self.model_name,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_map': self.best_val_map,
            'best_epoch': self.best_epoch,
            'config': {k: str(v) if isinstance(v, torch.device) else v
                       for k, v in self.config.items()},
        }

        # Save latest
        path = os.path.join(ckpt_dir, f'{self.model_name}_latest.pth')
        torch.save(checkpoint, path)

        # Save best
        if is_best:
            best_path = os.path.join(ckpt_dir, f'{self.model_name}_best.pth')
            torch.save(checkpoint, best_path)

    def load_checkpoint(self):
        """Load checkpoint if exists, return True if resumed."""
        ckpt_path = os.path.join(
            self.config['checkpoint_dir'], f'{self.model_name}_latest.pth'
        )
        if not os.path.exists(ckpt_path):
            return False

        print(f"🔄 Resuming {self.model_name} from checkpoint...")
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.best_val_map = checkpoint.get('best_val_map', 0.0)
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self.start_epoch = checkpoint['epoch'] + 1
        print(f"   Resumed at epoch {self.start_epoch}, best mAP: {self.best_val_map:.4f}")
        return True

    def train(self, train_loader, val_loader, resume=True):
        """Full training loop with early stopping and checkpointing."""
        if resume:
            self.load_checkpoint()

        print(f"\n{'='*60}")
        print(f"🏋️ Training {self.model_name}")
        print(f"   Epochs: {self.start_epoch} → {self.config['epochs']}")
        print(f"   Patience: {self.config['patience']}")
        print(f"{'='*60}\n")

        for epoch in range(self.start_epoch, self.config['epochs']):
            # Train
            train_loss, train_acc = self.train_one_epoch(train_loader)

            # Evaluate
            val_loss, val_acc, val_map, _, _, _ = self.evaluate(val_loader)

            # Update scheduler
            self.scheduler.step(val_map)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['val_map'].append(val_map)
            self.history['lr'].append(current_lr)

            # Check for best model
            is_best = val_map > self.best_val_map
            if is_best:
                self.best_val_map = val_map
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint (to Drive — persists across crashes)
            self.save_checkpoint(epoch, is_best=is_best)

            # Print progress
            best_marker = " 🏆 NEW BEST" if is_best else ""
            print(
                f"Epoch [{epoch+1:3d}/{self.config['epochs']}] "
                f"Loss: {train_loss:.4f}/{val_loss:.4f} "
                f"Acc: {train_acc:.3f}/{val_acc:.3f} "
                f"mAP: {val_map:.4f} "
                f"LR: {current_lr:.1e}"
                f"{best_marker}"
            )

            # Early stopping
            if self.patience_counter >= self.config['patience']:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}. "
                      f"Best mAP: {self.best_val_map:.4f} at epoch {self.best_epoch+1}")
                break

        # Load best model
        best_path = os.path.join(
            self.config['checkpoint_dir'], f'{self.model_name}_best.pth'
        )
        if os.path.exists(best_path):
            best_ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(best_ckpt['model_state_dict'])
            print(f"\n✅ Loaded best {self.model_name} (epoch {self.best_epoch+1}, mAP: {self.best_val_map:.4f})")

        return self.history

print("✅ Training engine defined")

# %% [markdown]
# ## Cell 8: Load Data

# %%
# ============================================================
# CELL 8: LOAD DATA (run this every time you reconnect)
# ============================================================

print("📥 Loading datasets...")
train_loader, val_loader, test_loader, train_dataset = create_dataloaders(CONFIG)

# Get class weights from training set
class_weights = getattr(train_dataset, 'class_weights', None)

# --- Dictionary to store results (persisted to Drive) ---
all_results = {}

print("✅ Data loaded. Now run Cells 8a, 8b, 8c below to train each model.")
print("   If Colab crashes, re-run Cells 1-8, then SKIP any model that's already done.")

# %% [markdown]
# ## Cell 8a: Train Baseline CNN

# %%
# ============================================================
# CELL 8a: TRAIN BASELINE CNN
# (Safe to re-run — skips if already fully trained)
# ============================================================

def train_and_evaluate_model(model_name, train_loader, val_loader, test_loader, config, class_weights):
    """
    Train a model, evaluate on test set, and save everything to Drive.
    If a completed checkpoint exists, it loads results instead of re-training.
    Returns the results dict.
    """
    # Check if this model already has saved test results on Drive
    results_path = os.path.join(config['checkpoint_dir'], f'{model_name}_test_results.pth')
    best_ckpt_path = os.path.join(config['checkpoint_dir'], f'{model_name}_best.pth')

    # If test results already exist AND best checkpoint exists, just reload
    if os.path.exists(results_path) and os.path.exists(best_ckpt_path):
        print(f"\n✅ {model_name.upper()} already trained! Loading saved results from Drive...")
        saved = torch.load(results_path, map_location=config['device'], weights_only=False)

        # Reload the model weights too (needed for inference later)
        model = get_model(model_name, config)
        best_ckpt = torch.load(best_ckpt_path, map_location=config['device'], weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'])
        model.eval()

        saved['model'] = model
        print(f"   Test Acc: {saved['test_acc']:.4f}, Test mAP: {saved['test_map']:.4f}")
        return saved

    # Otherwise, train (or resume from where we left off)
    print(f"\n{'#'*60}")
    print(f"# Training: {model_name.upper()}")
    print(f"{'#'*60}")

    model = get_model(model_name, config)
    engine = TrainingEngine(model, config, class_weights=class_weights)
    history = engine.train(train_loader, val_loader, resume=True)

    # Evaluate on test set
    test_loss, test_acc, test_map, preds, labels, probs = engine.evaluate(test_loader)

    results = {
        'model': model,
        'history': history,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_map': test_map,
        'preds': preds,
        'labels': labels,
        'probs': probs,
    }

    # Save test results to Drive (so we never have to re-train)
    save_data = {k: v for k, v in results.items() if k != 'model'}
    torch.save(save_data, results_path)

    print(f"\n📊 {model.name} Test Results:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test mAP: {test_map:.4f}")
    print(f"   💾 Results saved to Drive — safe even if Colab crashes now!")

    return results

# ----- TRAIN BASELINE -----
all_results['baseline'] = train_and_evaluate_model(
    'baseline', train_loader, val_loader, test_loader, CONFIG, class_weights
)

# 🛡️ BACKUP after Baseline training — downloads zip to your Mac
backup_and_download('after_baseline')

# %% [markdown]
# ## Cell 8b: Train Bi-LSTM (Primary Model)

# %%
# ============================================================
# CELL 8b: TRAIN BI-LSTM
# (Safe to re-run — skips if already fully trained)
# ============================================================
all_results['bilstm'] = train_and_evaluate_model(
    'bilstm', train_loader, val_loader, test_loader, CONFIG, class_weights
)

# 🛡️ BACKUP after BiLSTM training — downloads zip to your Mac
backup_and_download('after_bilstm')

# %% [markdown]
# ## Cell 8c: Train Transformer (Advanced Model)

# %%
# ============================================================
# CELL 8c: TRAIN TRANSFORMER
# (Safe to re-run — skips if already fully trained)
# ============================================================
all_results['transformer'] = train_and_evaluate_model(
    'transformer', train_loader, val_loader, test_loader, CONFIG, class_weights
)

# 🛡️ BACKUP after ALL training — downloads final zip to your Mac
backup_and_download('all_models_final')

print("\n" + "="*60)
print("✅ ALL 3 MODELS TRAINED AND EVALUATED!")
print("="*60)
for name, res in all_results.items():
    print(f"   {res['model'].name}: Acc={res['test_acc']:.4f}, mAP={res['test_map']:.4f}")

# %% [markdown]
# ## Cell 9: Visualization & Results

# %%
# ============================================================
# CELL 9: VISUALIZATION — Training Curves, Comparisons, Confusion Matrices
# ============================================================

results_dir = CONFIG['results_dir']
os.makedirs(results_dir, exist_ok=True)

# --- 1. Training Loss Curves ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Training Progress — All Models', fontsize=16, fontweight='bold')

for idx, (model_name, result) in enumerate(all_results.items()):
    ax = axes[idx]
    history = result['history']
    epochs_ran = range(1, len(history['train_loss']) + 1)

    ax.plot(epochs_ran, history['train_loss'], label='Train Loss', color='#e74c3c', linewidth=2)
    ax.plot(epochs_ran, history['val_loss'], label='Val Loss', color='#3498db', linewidth=2)
    ax.set_title(result['model'].name, fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'training_loss_curves.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Training loss curves saved")

# --- 2. Validation mAP Curves ---
fig, ax = plt.subplots(figsize=(10, 6))
colors = {'baseline': '#e74c3c', 'bilstm': '#2ecc71', 'transformer': '#9b59b6'}

for model_name, result in all_results.items():
    history = result['history']
    epochs_ran = range(1, len(history['val_map']) + 1)
    ax.plot(
        epochs_ran, history['val_map'],
        label=f"{result['model'].name} (best: {max(history['val_map']):.4f})",
        color=colors[model_name], linewidth=2.5
    )

ax.set_title('Validation mAP — Model Comparison', fontsize=16, fontweight='bold')
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('mAP', fontsize=13)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'validation_map_curves.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Validation mAP curves saved")

# --- 3. Model Comparison Bar Chart ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
metrics = ['test_acc', 'test_map', 'test_loss']
titles = ['Test Accuracy ↑', 'Test mAP ↑', 'Test Loss ↓']
color_list = ['#e74c3c', '#2ecc71', '#9b59b6']

for ax, metric, title in zip(axes, metrics, titles):
    values = [all_results[m][metric] for m in all_results]
    names = [all_results[m]['model'].name for m in all_results]

    bars = ax.bar(names, values, color=color_list, edgecolor='black', linewidth=0.5)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.3 if max(values) > 0 else 1)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.suptitle('Model Comparison — Test Set Performance', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Model comparison chart saved")

# --- 4. Confusion Matrices ---
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
class_names = CONFIG['event_classes']

for idx, (model_name, result) in enumerate(all_results.items()):
    ax = axes[idx]
    cm = confusion_matrix(result['labels'], result['preds'], labels=range(CONFIG['num_classes']))

    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax.set_title(result['model'].name, fontsize=13, fontweight='bold')

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    # Annotate cells
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = f'{cm_norm[i, j]:.2f}\n({cm[i, j]})'
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=8, color=color)

fig.colorbar(im, ax=axes, shrink=0.6)
plt.suptitle('Confusion Matrices (Normalized)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Confusion matrices saved")

# --- 5. Classification Report ---
print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORTS")
print("="*60)

for model_name, result in all_results.items():
    print(f"\n--- {result['model'].name} ---")
    print(classification_report(
        result['labels'], result['preds'],
        target_names=class_names,
        zero_division=0
    ))

# --- 6. Save Metrics to JSON ---
metrics_summary = {}
for model_name, result in all_results.items():
    metrics_summary[result['model'].name] = {
        'test_accuracy': float(result['test_acc']),
        'test_mAP': float(result['test_map']),
        'test_loss': float(result['test_loss']),
        'best_val_mAP': float(max(result['history']['val_map'])) if result['history']['val_map'] else 0,
        'total_epochs': len(result['history']['train_loss']),
    }

with open(os.path.join(results_dir, 'metrics_summary.json'), 'w') as f:
    json.dump(metrics_summary, f, indent=2)

print(f"\n✅ All visualizations saved to: {results_dir}")

# %% [markdown]
# ## Cell 10: Highlight Generation (Inference on a Single Match)

# %%
# ============================================================
# CELL 10: HIGHLIGHT GENERATION — Inference on a Single Match
# ============================================================

def knapsack_highlight_selection(events, max_duration_frames, fps=2):
    """
    Select the best set of highlight clips within a time budget,
    using the 0/1 Knapsack algorithm (from CA-SUM).

    Args:
        events: List of dicts with 'start', 'end', 'confidence', 'class'
        max_duration_frames: Maximum total frames for highlights
        fps: Frames per second
    Returns:
        Selected events list
    """
    if not events:
        return []

    n = len(events)
    weights = [e['end'] - e['start'] for e in events]
    values = [e['confidence'] for e in events]
    W = max_duration_frames

    # DP table
    K = [[0.0 for _ in range(W + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(W + 1):
            if weights[i-1] <= w:
                K[i][w] = max(
                    values[i-1] + K[i-1][w - weights[i-1]],
                    K[i-1][w]
                )
            else:
                K[i][w] = K[i-1][w]

    # Backtrack to find selected items
    selected_indices = []
    w = W
    for i in range(n, 0, -1):
        if K[i][w] != K[i-1][w]:
            selected_indices.insert(0, i - 1)
            w -= weights[i - 1]

    return [events[i] for i in selected_indices]

@torch.no_grad()
def generate_highlights(model, match_source, config, max_highlight_minutes=10):
    """
    Run inference on a single match and detect events.
    
    Args:
        model: Trained model
        match_source: Path to match directory (SoccerNet) OR direct path to a custom .npy file
        config: Configuration dict
        max_highlight_minutes: Maximum highlight duration in minutes
    """
    model.eval()
    device = config['device']
    results = {'half_1': [], 'half_2': []}

    is_direct_file = match_source.endswith('.npy')
    halves_to_process = [1] if is_direct_file else [1, 2]

    for half in halves_to_process:
        if is_direct_file:
            feat_path = match_source
        else:
            feat_path = os.path.join(match_source, f'{half}_ResNET_TF2_PCA512.npy')
            
        if not os.path.exists(feat_path):
            continue

        features = np.load(feat_path)
        num_frames = features.shape[0]
        seq_len = config['seq_len']

        # Create sequences
        all_probs = np.zeros((num_frames, config['num_classes']))
        frame_counts = np.zeros(num_frames)

        for start in range(0, num_frames - seq_len + 1):
            seq = torch.tensor(
                features[start:start + seq_len], dtype=torch.float32
            ).unsqueeze(0).to(device)

            logits = model(seq)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            center = start + seq_len // 2
            all_probs[center] += probs
            frame_counts[center] += 1

        # Average overlapping predictions
        valid = frame_counts > 0
        all_probs[valid] /= frame_counts[valid, np.newaxis]

        # Detect events (non-background predictions above threshold)
        bg_class = config['num_classes'] - 1
        event_mask = all_probs[:, :bg_class].max(axis=1) > 0.3

        # Group consecutive detections into events
        events = []
        in_event = False
        event_start = 0

        for i in range(num_frames):
            if event_mask[i] and not in_event:
                event_start = i
                in_event = True
            elif not event_mask[i] and in_event:
                best_class = all_probs[event_start:i, :bg_class].mean(axis=0).argmax()
                confidence = float(all_probs[event_start:i, best_class].mean())

                # Add padding around event
                pad = 30 * config['target_fps']  # ±30 seconds
                events.append({
                    'start': max(0, event_start - pad),
                    'end': min(num_frames, i + pad),
                    'event_frame': (event_start + i) // 2,
                    'class': config['event_classes'][best_class],
                    'class_idx': int(best_class),
                    'confidence': confidence,
                    'half': half,
                    'time_str': f"{(event_start // config['target_fps']) // 60}:{(event_start // config['target_fps']) % 60:02d}",
                })
                in_event = False

        # Handle event at end of half
        if in_event:
            i = num_frames
            best_class = all_probs[event_start:i, :bg_class].mean(axis=0).argmax()
            confidence = float(all_probs[event_start:i, best_class].mean())
            pad = 30 * config['target_fps']
            events.append({
                'start': max(0, event_start - pad),
                'end': min(num_frames, i + pad),
                'event_frame': (event_start + i) // 2,
                'class': config['event_classes'][best_class],
                'class_idx': int(best_class),
                'confidence': confidence,
                'half': half,
                'time_str': f"{(event_start // config['target_fps']) // 60}:{(event_start // config['target_fps']) % 60:02d}",
            })

        results[f'half_{half}'] = events

    # Combine all events
    all_events = results['half_1'] + results['half_2']

    # Apply knapsack to select best events within time budget
    max_frames = max_highlight_minutes * 60 * config['target_fps']
    selected_events = knapsack_highlight_selection(all_events, max_frames)

    # Sort by time
    selected_events.sort(key=lambda e: (e['half'], e['start']))

    return selected_events, all_events

# --- Run on test match OR Custom Video ---
test_matches = []
if os.path.exists(CONFIG['data_dir']):
    for root, dirs, files in os.walk(CONFIG['data_dir']):
        if 'Labels-v2.json' in files:
            test_matches.append(root)

# We ALWAYS want to use the best model we have loaded
best_model_name = max(all_results, key=lambda k: all_results[k]['test_map'])
best_model = all_results[best_model_name]['model']

# 1. Prioritize a Custom Uploaded Video (e.g. from Demo Day)
custom_feature_path = '/content/test_features.npy'

if os.path.exists(custom_feature_path):
    print(f"\n🎬 Generating highlights using YOUR CUSTOM VIDEO features... (Model: {best_model.name})")
    target_source = custom_feature_path
    
# 2. Fallback to SoccerNet/Synthetic dataset
elif test_matches:
    print(f"\n🎬 Generating highlights using a Dataset Match... (Model: {best_model.name})")
    target_source = test_matches[-1]
    
else:
    print("❌ No matches or custom features found to test on!")
    target_source = None

if target_source:
    selected_events, all_detected = generate_highlights(best_model, target_source, CONFIG)

    print(f"\n📍 Detected {len(all_detected)} total events, selected {len(selected_events)} for highlights:")
    for evt in selected_events:
        print(f"   Half {evt['half']} — {evt['time_str']} — {evt['class']} (conf: {evt['confidence']:.3f})")

    # Save events to JSON
    output_path = os.path.join(CONFIG['outputs_dir'], 'detected_events.json')
    with open(output_path, 'w') as f:
        json.dump({'selected': selected_events, 'all_detected': all_detected}, f, indent=2)
    print(f"\n✅ Events saved to {output_path}")

    # --- Visualize event timeline ---
    fig, ax = plt.subplots(figsize=(16, 4))
    event_colors = {'Goal': '#e74c3c', 'Cards': '#f1c40f', 'Substitution': '#3498db'}

    for evt in selected_events:
        color = event_colors.get(evt['class'], '#95a5a6')
        start_min = evt['start'] / CONFIG['target_fps'] / 60
        end_min = evt['end'] / CONFIG['target_fps'] / 60
        half_offset = 0 if evt['half'] == 1 else 45
        ax.barh(0, end_min - start_min, left=start_min + half_offset,
                height=0.6, color=color, edgecolor='black', linewidth=0.5, alpha=0.8)
        ax.text(start_min + half_offset + (end_min - start_min) / 2, 0,
                evt['class'], ha='center', va='center', fontsize=8, fontweight='bold')

    ax.set_xlim(0, 95)
    ax.set_yticks([])
    ax.set_xlabel('Match Time (minutes)', fontsize=13)
    ax.set_title('Detected Highlight Events Timeline', fontsize=16, fontweight='bold')
    ax.axvline(x=45, color='gray', linestyle='--', linewidth=1, label='Half Time')

    patches = [mpatches.Patch(color=c, label=l) for l, c in event_colors.items()]
    ax.legend(handles=patches, loc='upper right')
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['results_dir'], 'event_timeline.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Event timeline saved")
else:
    print("⚠️  No match data found. Skipping highlight generation.")
    print("   Download SoccerNet data first (Cell 4)")

# %% [markdown]
# ## Cell 10b: End-to-End Raw Video Feature Extraction

# %%
# ============================================================
# CELL 10b: RAW VIDEO → FEATURES (End-to-End Pipeline)
# ============================================================
import cv2
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def extract_features_from_video(video_path, output_path, target_fps=2):
    """
    Extracts ResNet-152 features from video, then applies PCA to reduce
    from 2048-dim to 512-dim — closely matching SoccerNet's feature format.
    """
    print(f"🎬 Processing video: {video_path}")
    if not os.path.exists(video_path):
        print("❌ Video file not found.")
        return False

    device = CONFIG['device']
    from sklearn.decomposition import PCA

    # ---- ResNet-152 instead of ResNet-18 ----
    # SoccerNet uses ResNet-152 (TF2). We use ResNet-152 (PyTorch).
    # Both are pre-trained on ImageNet → similar feature representations.
    print("📦 Loading ResNet-152 (matches SoccerNet's feature extractor)...")
    resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    modules = list(resnet.children())[:-1]
    feature_extractor = nn.Sequential(*modules).to(device)
    feature_extractor.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("❌ Could not read video FPS.")
        return False

    frame_interval = max(1, int(round(fps / target_fps)))
    raw_features = []

    count = 0
    saved_count = 0
    print("⏳ Extracting ResNet-152 features (2048-dim)... This will take a while.")
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                input_tensor = preprocess(img).unsqueeze(0).to(device)

                feat = feature_extractor(input_tensor)
                feat = feat.view(-1).cpu().numpy()  # 2048-dim vector
                raw_features.append(feat)
                saved_count += 1

                if saved_count % 500 == 0:
                    print(f"   Extracted {saved_count} frames "
                          f"({(saved_count/target_fps)/60:.1f} min of video)...")
            count += 1

    cap.release()

    raw_features = np.array(raw_features)  # Shape: (N, 2048)
    print(f"✅ Raw features shape: {raw_features.shape}")

    # ---- Apply PCA 2048 → 512 ----
    print("🔄 Applying PCA: 2048-dim → 512-dim (matching SoccerNet format)...")
    if raw_features.shape[0] < 512:
        pca = PCA(n_components=raw_features.shape[0])
        features_reduced = pca.fit_transform(raw_features)
        features_512 = np.pad(features_reduced, ((0,0), (0, 512 - raw_features.shape[0])))
    else:
        pca = PCA(n_components=512)
        features_512 = pca.fit_transform(raw_features)

    print(f"✅ Final features shape: {features_512.shape}")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    np.save(output_path, features_512.astype(np.float32))
    print(f"💾 Saved features to: {output_path}")
    return True

print("✅ End-to-End Feature Extractor (ResNet-152 + PCA) defined.")

# %% [markdown]
# ## Cell 10c: Create Final Highlight Video Output

# %%
# ============================================================
# CELL 10c: STITCH HIGHLIGHTS INTO FINAL MP4
# ============================================================
def create_highlight_video(video_path, events, output_path):
    """Cuts the video using detected events and merges them together."""
    try:
        try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips
    except ImportError:
        from moviepy import VideoFileClip, concatenate_videoclips
    except ImportError:
        print("❌ moviepy is not installed. Run Cell 2.")
        return
        
    if not os.path.exists(video_path):
        print("❌ Source video not found.")
        return
        
    print(f"\n✂️ Cutting {len(events)} highlights from {video_path}...")
    try:
        video = VideoFileClip(video_path)
        clips = []
        
        for i, evt in enumerate(events):
            # Convert frame indices to seconds (events are detected at target_fps)
            target_fps = CONFIG['target_fps']
            start_sec = max(0, evt['start'] / target_fps)
            end_sec = min(video.duration, evt['end'] / target_fps)
            
            if end_sec <= start_sec: continue
            
            print(f"   Clip {i+1}: {start_sec:.1f}s --> {end_sec:.1f}s ({evt['class']})")
            if hasattr(video, 'subclip'):
            clip = video.subclip(start_sec, end_sec)
        else:
            clip = video.subclipped(start_sec, end_sec)
            clips.append(clip)
            
        if not clips:
            print("❌ No valid clips extracted.")
            return
            
        print("🔄 Stitching clips together (this may take a while depending on video length)...")
        final_video = concatenate_videoclips(clips)
        final_video.write_videofile(
            output_path, 
            codec="libx264", 
            audio_codec="aac",
            logger=None # Suppresses massive progress bars in Colab
        )
        print(f"🎉 SUCCESS! Final highlight reel saved to: {output_path}")
        
        if ON_COLAB:
            from google.colab import files
            print("⬇️ Downloading the highlight video to your Mac...")
            files.download(output_path)
        
    except Exception as e:
        print(f"❌ Error during video processing: {e}")

print("✅ Highlight video creator defined.")
print("   Usage:")
print("   1. run model to get events -> selected_events, _ = generate_highlights(...)")
print("   2. create_highlight_video('/content/my_match.mp4', selected_events, '/content/final_highlights.mp4')")

# %% [markdown]
# ## Cell 11: Backup & Export

# %%
# ============================================================
# CELL 11: ZIP BACKUP — Never Lose Your Work
# ============================================================
import zipfile

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
zip_path = os.path.join(PROJECT_DIR, f'backup_{timestamp}.zip')

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Backup checkpoints (best models only to save space)
    for f in glob.glob(os.path.join(CONFIG['checkpoint_dir'], '*_best.pth')):
        zipf.write(f, f'checkpoints/{os.path.basename(f)}')

    # Backup results
    for f in glob.glob(os.path.join(CONFIG['results_dir'], '*')):
        zipf.write(f, f'results/{os.path.basename(f)}')

    # Backup outputs
    for f in glob.glob(os.path.join(CONFIG['outputs_dir'], '*')):
        zipf.write(f, f'outputs/{os.path.basename(f)}')

zip_size = os.path.getsize(zip_path) / (1024 * 1024)
print(f"✅ Backup saved: {zip_path}")
print(f"   Size: {zip_size:.1f} MB")

# Download to your Mac (Colab only)
if ON_COLAB:
    print("📥 Download the backup to your Mac:")
    print(f"   from google.colab import files; files.download('{zip_path}')")

# %% [markdown]
# ## Cell 12: Gradio UI (Optional — For Presentation Demo)

# %%
# ============================================================
# CELL 12: GRADIO UI — Interactive Demo for Presentation
# ============================================================

def launch_gradio_demo():
    """Launch a Gradio web interface for interactive highlight detection."""
    try:
        import gradio as gr
    except ImportError:
        os.system('pip install gradio -q')
        import gradio as gr

    # Load best model
    best_model_name = max(all_results, key=lambda k: all_results[k]['test_map'])
    model = all_results[best_model_name]['model']
    model.eval()

    # Find available test matches
    available_matches = []
    if os.path.exists(CONFIG['data_dir']):
        for root, dirs, files in os.walk(CONFIG['data_dir']):
            if 'Labels-v2.json' in files:
                match_name = os.path.basename(root)
                available_matches.append((match_name, root))

    match_choices = [name for name, _ in available_matches[:20]]  # Limit to 20

    def detect_events(match_name, threshold):
        """Run detection on selected match."""
        # Find match path
        match_path = None
        for name, path in available_matches:
            if name == match_name:
                match_path = path
                break

        if not match_path:
            return "Match not found", None, None

        selected, all_events = generate_highlights(model, match_path, CONFIG)

        # Filter by threshold
        selected = [e for e in selected if e['confidence'] >= threshold]

        # Format results
        results_text = f"**Model**: {all_results[best_model_name]['model'].name}\n"
        results_text += f"**Match**: {match_name}\n"
        results_text += f"**Events detected**: {len(selected)}\n\n"
        results_text += "| Time | Event | Confidence |\n|---|---|---|\n"
        for evt in selected:
            results_text += f"| Half {evt['half']} - {evt['time_str']} | {evt['class']} | {evt['confidence']:.3f} |\n"

        # Create timeline plot
        fig, ax = plt.subplots(figsize=(14, 3))
        event_colors = {'Goal': '#e74c3c', 'Cards': '#f1c40f', 'Substitution': '#3498db'}
        for evt in selected:
            color = event_colors.get(evt['class'], '#95a5a6')
            start_min = evt['start'] / CONFIG['target_fps'] / 60
            end_min = evt['end'] / CONFIG['target_fps'] / 60
            half_offset = 0 if evt['half'] == 1 else 45
            ax.barh(0, end_min - start_min, left=start_min + half_offset,
                    height=0.6, color=color, alpha=0.8, edgecolor='black')
        ax.set_xlim(0, 95)
        ax.set_yticks([])
        ax.set_xlabel('Minutes')
        ax.set_title('Event Timeline')
        ax.axvline(x=45, color='gray', linestyle='--')
        plt.tight_layout()

        return results_text, fig

    # Build Gradio interface
    demo = gr.Interface(
        fn=detect_events,
        inputs=[
            gr.Dropdown(choices=match_choices, label="🏟️ Select Match", value=match_choices[0] if match_choices else None),
            gr.Slider(minimum=0.1, maximum=0.95, value=0.5, step=0.05, label="📊 Confidence Threshold"),
        ],
        outputs=[
            gr.Markdown(label="📋 Detection Results"),
            gr.Plot(label="📍 Event Timeline"),
        ],
        title="🏟️ AI Football Highlight Detection System",
        description="Select a match and adjust the confidence threshold to detect highlights using our trained CNN + Bi-LSTM / Transformer model.",
        theme=gr.themes.Soft(),
    )

    demo.launch(share=True)  # Creates a public URL

# Uncomment to launch:
# launch_gradio_demo()

# %% [markdown]
# ## Cell 13: Final Summary

# %%
# ============================================================
# CELL 13: FINAL SUMMARY
# ============================================================
print("="*60)
print("🏟️ FINAL RESULTS SUMMARY")
print("="*60)

print(f"\n{'Model':<20} {'Test Acc':>10} {'Test mAP':>10} {'Test Loss':>10}")
print("-"*55)
for model_name, result in all_results.items():
    name = result['model'].name
    print(f"{name:<20} {result['test_acc']:>10.4f} {result['test_map']:>10.4f} {result['test_loss']:>10.4f}")

best_name = max(all_results, key=lambda k: all_results[k]['test_map'])
print(f"\n🏆 Best Model: {all_results[best_name]['model'].name} (mAP: {all_results[best_name]['test_map']:.4f})")

print(f"\n📁 Results saved to: {CONFIG['results_dir']}")
print(f"💾 Checkpoints at: {CONFIG['checkpoint_dir']}")
print(f"📦 Backup at: {PROJECT_DIR}")
print("\n✅ Project complete! Ready for presentation.")
