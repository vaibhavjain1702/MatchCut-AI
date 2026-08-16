# 🏆 THE COMPLETE PROJECT GUIDE — Every Line, Every Concept, Every Question

> **Your interview is tomorrow. This document covers EVERYTHING.**
> Read it top to bottom. After this, no question can stump you.

---

## TABLE OF CONTENTS

1. [Project Overview — The 30-Second Elevator Pitch](#section-1)
2. [System Architecture — The Big Picture](#section-2)
3. [Cell-by-Cell Code Walkthrough](#section-3)
   - Cell 1: Setup & Imports
   - Cell 2/2b: Dependencies & Backup
   - Cell 3: Configuration (Every Hyperparameter Explained)
   - Cell 4: SoccerNet Data Download
   - Cell 5: Dataset Class (The Data Pipeline)
   - Cell 6: Model Definitions (All 3 Models)
   - Cell 7: Training Engine
   - Cells 8/8a/8b/8c: Training Execution
   - Cell 9: Visualization
   - Cells 10/10b/10c: Highlight Generation Pipeline
   - Cell 12: Gradio UI
4. [The Streamlit App (app.py)](#section-4)
5. [Key Deep Learning Concepts Explained Simply](#section-5)
6. [40+ Interview Questions with Model Answers](#section-6)

---

# 1. PROJECT OVERVIEW — THE 30-SECOND ELEVATOR PITCH <a name="section-1"></a>

> "I built an AI system that automatically generates highlight reels from full-length football match videos. It uses a two-stage deep learning pipeline: first, a pre-trained ResNet-152 CNN extracts visual features from each frame. Then, a Bidirectional LSTM processes sequences of these features to detect key events — Goals, Cards, and Substitutions. Finally, a Knapsack optimization algorithm selects the best clips within a time budget and stitches them into a final highlight video. The system was trained on 60 matches from the SoccerNet dataset and achieved 35.1% mAP, which is approximately 50% of the state-of-the-art while using only 12% of the available training data on a single free Google Colab GPU."

---

# 2. SYSTEM ARCHITECTURE — THE BIG PICTURE <a name="section-2"></a>

```
Raw Match Video (MP4, 90 minutes)
        │
        ▼
┌──────────────────────────────┐
│  STAGE 1: SPATIAL FEATURES   │
│  ResNet-152 (pre-trained     │
│  on ImageNet)                │
│  Extracts 2048-dim vector    │
│  per frame at 2 FPS          │
│  Then PCA reduces to 512-dim │
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│  STAGE 2: TEMPORAL MODELING  │
│  BiLSTM (2 layers, 256       │
│  hidden units, bidirectional)│
│  Reads 30-frame windows      │
│  (15 seconds) and classifies │
│  the CENTER frame            │
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│  STAGE 3: EVENT SELECTION    │
│  Threshold at 0.3            │
│  Group consecutive frames    │
│  Add ±30 second padding      │
│  0/1 Knapsack DP algorithm   │
│  (maximize confidence under  │
│  10-minute time budget)      │
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│  STAGE 4: VIDEO OUTPUT       │
│  MoviePy cuts clips from     │
│  original video              │
│  Concatenates them           │
│  H.264 codec + AAC audio     │
│  → Final highlights.mp4      │
└──────────────────────────────┘
```

### Why Two Stages Instead of End-to-End?

End-to-end would mean: raw pixels → CNN → temporal model → event labels, all trained together with backpropagation flowing through both. This requires holding the entire CNN + LSTM in GPU memory simultaneously with all intermediate activations for gradient computation. That needs **40+ GB of VRAM**. Colab's free T4 has only **15 GB**.

The two-stage approach:
- **Stage 1 is done ONCE** — extract features, save as `.npy` files on disk
- **Stage 2 trains iteratively** — loads the small 512-dim vectors, not raw images
- This reduces memory from ~40GB to ~2GB during training

---

# 3. CELL-BY-CELL CODE WALKTHROUGH <a name="section-3"></a>

The main file is [Football_Highlight_Detection.py](file:///Users/vaibhav/Study/Sem%206/Deep%20Learning/Project/Football_Highlight_Detection.py) (1892 lines). It's structured as a Jupyter notebook exported to `.py` format. Each `# %%` marker is a notebook cell.

---

## CELL 1: SETUP & IMPORTS (Lines 1–78)

### Lines 25–35: Standard Python Imports
```python
import os           # File system operations (paths, directories)
import sys          # System-level operations
import json         # Read/write JSON files (labels, config)
import glob         # File pattern matching (find all .pth files)
import random       # Random number generation (shuffling, seeding)
import shutil       # File copy/move operations
import datetime     # Timestamps for backup filenames
import warnings     # Suppress unnecessary warnings
import numpy as np  # Numerical computing (arrays, matrix operations)
from pathlib import Path           # Modern path handling
from collections import defaultdict  # Dictionary with default values
```

### Lines 37–40: PyTorch Core Imports
```python
import torch                              # The deep learning framework
import torch.nn as nn                      # Neural network layers (Linear, LSTM, Conv2d, etc.)
import torch.optim as optim                # Optimizers (Adam, SGD, etc.)
from torch.utils.data import Dataset, DataLoader  # Data pipeline abstractions
```
- **`torch`**: The core library. Provides tensors (GPU-accelerated arrays), automatic differentiation, and neural network building blocks.
- **`nn.Module`**: Every neural network you build inherits from this. It tracks parameters, handles GPU movement, and enables `model.train()`/`model.eval()` mode switching.
- **`Dataset`**: An abstract class. You implement `__getitem__(idx)` (return one sample) and `__len__()` (total samples). PyTorch's DataLoader then handles batching, shuffling, and parallel loading.
- **`DataLoader`**: Wraps a Dataset. It takes samples from `__getitem__`, groups them into batches of size `batch_size`, optionally shuffles them every epoch, and loads them in parallel using `num_workers` background processes.

### Lines 42–49: Visualization & Metrics
```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no display needed on servers)
import matplotlib.pyplot as plt           # Plotting library
import matplotlib.patches as mpatches     # Custom legend patches
from sklearn.metrics import (
    confusion_matrix,          # NxN matrix of true vs predicted labels
    classification_report,     # Precision, recall, F1 per class
    precision_recall_curve,    # PR curve for threshold analysis
    average_precision_score    # Area under the PR curve (= AP)
)
```
- **`matplotlib.use('Agg')`**: Critical for servers/Colab. Without this, matplotlib tries to open a GUI window, which crashes on headless systems. `'Agg'` renders plots to files/memory instead.
- **`average_precision_score`**: THIS is how we compute mAP. For each class, it creates a binary "is this class?" label, ranks all samples by predicted probability, and computes the area under the precision-recall curve.

### Lines 51: Suppress Warnings
```python
warnings.filterwarnings('ignore')
```
Suppresses PyTorch deprecation warnings and sklearn convergence warnings that clutter Colab output.

### Lines 53–68: Environment Detection (Colab vs Local)
```python
try:
    from google.colab import drive    # This import ONLY succeeds on Colab
    ON_COLAB = True
    drive.mount('/content/drive')     # Mount Google Drive as a filesystem
    PROJECT_DIR = '/content/drive/MyDrive/DL_Project'
except ImportError:
    ON_COLAB = False                  # Running on your local Mac
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
```
**Why Google Drive?** Colab VMs are **ephemeral** — if you disconnect or the 12-hour session limit expires, ALL local files are deleted. By saving to Drive, your checkpoints and results persist across sessions. This is why `PROJECT_DIR` points to Drive on Colab but to the local directory when running on your Mac.

### Lines 70–72: Create Directory Structure
```python
for d in ['data', 'checkpoints', 'results', 'outputs']:
    os.makedirs(os.path.join(PROJECT_DIR, d), exist_ok=True)
```
Creates 4 subdirectories if they don't exist. `exist_ok=True` prevents errors if they already exist.
- **`data/`**: SoccerNet features and labels
- **`checkpoints/`**: Saved model weights (`.pth` files)
- **`results/`**: Training plots and metrics
- **`outputs/`**: Generated highlight videos and event JSONs

### Lines 74–78: GPU Detection
```python
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
- **`'cuda'`**: NVIDIA GPU available (Colab's T4 with 15GB VRAM). Training is 10-50x faster on GPU.
- **`'cpu'`**: No GPU. Training will be extremely slow but still works.
- All tensors and models must be moved to the same device: `tensor.to(DEVICE)`, `model.to(DEVICE)`.

---

## CELL 2: INSTALL DEPENDENCIES (Lines 83–89)

```python
if ON_COLAB:
    os.system('pip install SoccerNet moviepy -q')
```
- **`SoccerNet`**: Official Python API for downloading and loading the SoccerNet dataset.
- **`moviepy`**: Video editing library. Used to cut clips from the original video and stitch them together.
- **`-q`**: Quiet mode, suppresses pip output spam.
- Only runs on Colab because local environments should have dependencies pre-installed via `requirements.txt`.

---

## CELL 2b: BACKUP & RESTORE UTILITIES (Lines 94–201)

### `backup_and_download()` (Lines 100–150)
```python
def backup_and_download(label="checkpoint"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    zip_name = f'backup_{label}_{timestamp}.zip'
```
Creates a timestamped zip file containing ALL checkpoints, results, and outputs.

```python
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Iterate over checkpoints/, results/, outputs/ and add all files
```
- **`ZIP_DEFLATED`**: Compresses files (vs just storing them). Reduces backup size significantly.
- After zipping, it calls `google.colab.files.download(zip_path)` which triggers a browser download to your Mac.

**Why this exists:** Colab sessions crash randomly. After every model finishes training, a backup is downloaded to your Mac. If Colab crashes mid-training, you can restore from the backup and skip already-trained models.

### `restore_from_backup()` (Lines 152–197)
The reverse operation. Upload a previously downloaded backup zip to Colab, and this function extracts checkpoints back into the correct directories. The training code checks for existing checkpoints and skips already-trained models.

---

## CELL 3: CONFIGURATION — EVERY HYPERPARAMETER (Lines 206–286)

This is a single Python dictionary (`CONFIG`) that controls the entire pipeline. **You must know every value and why.**

### Data Parameters

| Parameter | Value | Explanation |
|---|---|---|
| `feature_dim: 512` | The dimensionality of each frame's feature vector. SoccerNet provides ResNet-152 features reduced from 2048 to 512 dimensions using PCA. We match this exactly. |
| `target_fps: 2` | SoccerNet extracts features at 2 frames per second. A 45-minute half = 45×60×2 = **5,400 frames**. |
| `seq_len: 30` | The sliding window size. 30 frames at 2 FPS = **15 seconds of video context**. The model sees 15 seconds and predicts what's happening at the middle. |
| `seq_stride: 15` | How far the window slides each step. 15 = 50% overlap. Each frame gets predictions from ~2 overlapping windows, which are averaged for smoother output. |
| `max_matches_train: 60` | Number of matches for training. Limited by Colab's 112GB disk. Each match is ~20MB of features. 60 matches ≈ 1.2GB. |
| `max_matches_val: 20` | Number of matches for validation (used during training to check mAP). |
| `max_matches_test: 20` | Number of matches for final testing (never seen during training). |

### Event Classes

```python
'event_classes': ['Goal', 'Cards', 'Substitution', 'Background'],
'num_classes': 4,
```
SoccerNet has 17 event types (Goal, Penalty, Yellow Card, Red Card, Corner, Free Kick, etc.). We merged them into **4 classes** because with only 60 matches, most of the 17 classes have fewer than 10 examples each — far too few to train on. Merging ensures each class has hundreds of examples.

```python
'label_map': {
    'Goal': 'Goal',
    'Penalty': 'Goal',                    # Penalties are goal-like events
    'Own goal - Loss of Ball Rec': 'Goal', # Own goals are still goals
    'Yellow card': 'Cards',
    'Red card': 'Cards',
    'Yellow card -> Red card': 'Cards',    # Double yellow = red
    'Substitution': 'Substitution',
    # Everything else → Background (ignored)
},
```

```python
'event_window': 10,  # ±10 frames = ±5 seconds around each event
```
When a goal happens at frame 2850, we don't just label that ONE frame. We label frames 2840–2860 (±10 frames = ±5 seconds). Without this, positive examples would be just 1 frame out of 5400, making the dataset impossibly sparse.

### Model Parameters

| Parameter | Value | Explanation |
|---|---|---|
| `hidden_dim: 256` | LSTM hidden state size. Each direction produces a 256-dim vector. Bidirectional = 256×2 = 512-dim output. Too small (64) → underfits. Too large (512) → overfits on 60 matches. |
| `num_heads: 8` | Transformer attention heads. With 512-dim input, each head processes 512/8 = 64 dimensions. This is the standard configuration from the "Attention Is All You Need" paper. |
| `num_layers: 2` | Both LSTM and Transformer use 2 layers of depth. Deeper networks need more data to avoid overfitting; 2 layers is optimal for our 60-match dataset. |
| `ff_dim: 1024` | Transformer feedforward dimension. Each Transformer layer has: Attention → FeedForward(512→1024→512). The expansion factor of 2x (512→1024) is standard. |
| `dropout: 0.4` | 40% dropout — aggressive, but necessary. With only 60 matches, the model easily memorizes the training data. Dropout randomly zeroes 40% of neurons during training, forcing the network to build redundant pathways and generalize better. |

### Training Parameters

| Parameter | Value | Explanation |
|---|---|---|
| `lr: 5e-4` | Learning rate for Adam optimizer. 5×10⁻⁴ is the "sweet spot" for Adam on classification tasks. Too high → training oscillates and doesn't converge. Too low → training is too slow. |
| `weight_decay: 1e-5` | L2 regularization. Adds a penalty of `1e-5 × ||weights||²` to the loss. Prevents weights from growing too large, reducing overfitting. Very small because Adam's adaptive rates already provide implicit regularization. |
| `batch_size: 32` | 32 sequences per gradient update. Each sequence is 30×512 = 15,360 floats = ~60KB. A batch of 32 = ~2MB. Fits easily in the T4's 15GB VRAM with room for model parameters and gradients. |
| `epochs: 80` | Maximum number of training epochs. In practice, early stopping triggers at epoch 15–25. |
| `patience: 15` | Early stopping patience. If validation mAP doesn't improve for 15 consecutive epochs, training stops. Prevents wasting time on diminishing returns. |
| `clip_grad: 5.0` | Gradient clipping. If the L2 norm of all gradients exceeds 5.0, they are rescaled proportionally. Prevents "exploding gradients" — a known problem with LSTMs where gradients accumulate multiplicatively through time steps and become NaN. |
| `reg_factor: 0.15` | Length regularization target. The model should predict roughly 15% of frames as events. |
| `reg_lambda: 0.5` | Weight of the length regularization loss term. Total loss = CrossEntropy + 0.5 × LengthRegularization. |

### Reproducibility

```python
torch.manual_seed(CONFIG['seed'])        # PyTorch random seed
np.random.seed(CONFIG['seed'])           # NumPy random seed
random.seed(CONFIG['seed'])              # Python random seed
torch.cuda.manual_seed_all(CONFIG['seed'])  # GPU random seed
```
Setting seed to 42 ensures the same random initialization, data shuffling, and dropout masks every time. This makes experiments reproducible — running the same code twice produces the same results.

---

## CELL 4: DOWNLOAD SOCCERNET DATA (Lines 291–363)

```python
def download_soccernet_subset():
    from SoccerNet.Downloader import SoccerNetDownloader
    downloader = SoccerNetDownloader(LocalDirectory=local_dir)
    downloader.downloadGames(
        files=["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy", "Labels-v2.json"],
        split=["test"]
    )
```

**What it downloads per match (3 files):**
1. **`1_ResNET_TF2_PCA512.npy`** — First half features. A NumPy array of shape `(~5400, 512)`. Each row is one frame's 512-dimensional feature vector, extracted by the SoccerNet team using ResNet-152 (TensorFlow 2) + PCA.
2. **`2_ResNET_TF2_PCA512.npy`** — Second half features. Same format.
3. **`Labels-v2.json`** — Event annotations. Contains a list of annotations, each with:
   - `gameTime`: e.g., `"1 - 23:45"` (half 1, 23 minutes 45 seconds)
   - `label`: e.g., `"Goal"`, `"Yellow card"`, `"Substitution"`

**Why the "test" split?** SoccerNet has 3 splits: train (300 matches), validation (100), test (100). The train split is too large for Colab's 112GB disk (~60GB of features). The test split is the smallest. We download it and re-split internally into 60/20/20 (train/val/test).

After downloading, the function copies selected matches to Google Drive and cleans up the temporary files from Colab's local disk.

---

## CELL 5: DATASET CLASS (Lines 368–580)

This is `SoccerNetHighlightDataset`, a PyTorch `Dataset` subclass. **This is one of the most important parts to understand.**

### `__init__()` (Lines 379–426)
```python
def __init__(self, data_dir, split='train', config=None):
    self.sequences = []    # List of tensors, each shape (30, 512)
    self.labels = []       # List of integers (0=Goal, 1=Cards, 2=Sub, 3=Background)
    self.match_info = []   # Metadata for debugging
```

1. Finds all match directories that contain both `.npy` features AND `Labels-v2.json`
2. Shuffles them with a fixed seed (for reproducibility)
3. Splits: first 60 → train, next 20 → val, last 20 → test
4. Processes each match by calling `_process_match()`

### `_process_match()` (Lines 440–509) — THE CORE DATA PIPELINE

**Step 1: Load features**
```python
features = np.load(feat_path)  # Shape: (num_frames, 512)
```
For a 45-minute half at 2 FPS: shape is approximately `(5400, 512)`.

**Step 2: Initialize all labels as Background**
```python
frame_labels = np.full(num_frames, self.config['num_classes'] - 1, dtype=np.int64)
```
Creates an array of length `num_frames`, filled with `3` (Background). This is the default for every frame — nothing is happening in most of a football match.

**Step 3: Parse event annotations**
```python
for ann in labels_data.get('annotations', []):
    game_time = ann.get('gameTime', '')     # e.g., "1 - 23:45"
    label = ann.get('label', '')            # e.g., "Yellow card"
    
    half_str, time_str = game_time.split(' - ')  # "1", "23:45"
    ann_half = int(half_str.strip())              # 1
    minutes, seconds = time_str.strip().split(':')  # "23", "45"
    total_seconds = int(minutes) * 60 + int(seconds)  # 1425
    frame_idx = int(total_seconds * self.config['target_fps'])  # 1425 * 2 = 2850
```
This converts the human-readable timestamp `"1 - 23:45"` into a frame index `2850`.

**Step 4: Label a window around the event**
```python
mapped_label = self.config['label_map'].get(label, None)  # "Yellow card" → "Cards"
class_idx = self.config['event_classes'].index(mapped_label)  # "Cards" → 1

window = self.config['event_window']  # 10
start = max(0, frame_idx - window)     # max(0, 2850 - 10) = 2840
end = min(num_frames, frame_idx + window + 1)  # min(5400, 2861) = 2861
frame_labels[start:end] = class_idx    # Frames 2840-2860 all labeled as "Cards"
```
Instead of labeling just frame 2850, we label a ±10 frame window (±5 seconds). This is critical because:
1. Event timestamps in SoccerNet are approximate (±2 seconds accuracy)
2. Events aren't instantaneous — a yellow card involves the referee approaching, showing the card, and the player reacting
3. Without this, we'd have ~5 positive frames out of 5400 — far too few for the model to learn

**Step 5: Create sliding window sequences**
```python
for start_idx in range(0, num_frames - seq_len, stride):
    end_idx = start_idx + seq_len          # start + 30
    seq_features = features[start_idx:end_idx]  # Shape: (30, 512)
    center_label = seq_labels[seq_len // 2]     # Label of frame 15 (the CENTER)
    
    self.sequences.append(torch.tensor(seq_features, dtype=torch.float32))
    self.labels.append(center_label)
```
This is the sliding window approach:
- Take 30 consecutive frames (15 seconds at 2 FPS)
- The **label** is whatever class the CENTER frame (frame 15) belongs to
- Slide the window forward by `stride=15` frames (50% overlap)
- For a 5400-frame half with stride 15: creates ~(5400-30)/15 ≈ **358 sequences per half**
- 60 matches × 2 halves × 358 = **~43,000 training sequences total**

### `_compute_class_weights()` (Lines 531–549)
```python
weights[i] = total / (self.config['num_classes'] * label_counts[i])
```
**The Class Imbalance Problem:**
- In 43,000 sequences: ~40,000 are Background (93%), ~1,000 are Goals (2.3%), ~1,000 are Cards (2.3%), ~1,000 are Substitutions (2.3%)
- Without weighting, the model would just predict "Background" for everything and get 93% accuracy while detecting ZERO events
- **Inverse frequency weighting**: Background gets weight ~0.26, Goals get ~12.5
- This means: if the model misclassifies a Goal as Background, the loss is **50x larger** than misclassifying Background correctly
- These weights are passed to `nn.CrossEntropyLoss(weight=class_weights)`

### `__getitem__()` and `__len__()` (Lines 551–555)
```python
def __len__(self):
    return len(self.sequences)

def __getitem__(self, idx):
    return self.sequences[idx], self.labels[idx]
```
Standard PyTorch Dataset interface. The DataLoader calls these to build batches.

### `create_dataloaders()` (Lines 557–578)
```python
train_loader = DataLoader(
    train_dataset, batch_size=32,
    shuffle=True,       # Randomize order every epoch (prevents memorization)
    num_workers=2,      # 2 background processes for data loading
    pin_memory=True,    # Pre-allocate GPU memory for faster transfer
    drop_last=True      # Drop incomplete final batch
)
```
- **`shuffle=True`** (train only): Randomizes sample order each epoch. Prevents the model from memorizing the order.
- **`num_workers=2`**: Loads data in 2 parallel background processes while the GPU trains on the current batch. Keeps the GPU busy.
- **`pin_memory=True`**: Allocates data in page-locked CPU memory, enabling faster CPU→GPU transfer.
- **`drop_last=True`** (train only): If the last batch has fewer than 32 samples (e.g., 7), it's dropped. This prevents BatchNorm statistics from being unreliable on tiny batches and keeps the length regularization term stable.

---

## CELL 6: MODEL DEFINITIONS (Lines 586–806) — THE 3 ARCHITECTURES

### MODEL A: BaselineCNN (Lines 593–631)

**Purpose:** An ablation baseline with NO temporal modeling. Proves that temporal context matters.

```python
class BaselineCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = "CNN_Baseline"
```

**Architecture:**
```
Input: (batch_size, 30, 512)
   │
   ▼ Take ONLY the center frame
(batch_size, 512)
   │
   ▼ nn.Linear(512, 256)  — Fully connected layer: 512 inputs → 256 outputs
   ▼ nn.ReLU()             — Activation: max(0, x). Introduces non-linearity.
   ▼ nn.Dropout(0.4)       — Randomly zero 40% of neurons during training
   ▼ nn.LayerNorm(256)     — Normalize activations across features (stabilizes training)
   │
   ▼ nn.Linear(256, 128)  — Second hidden layer
   ▼ nn.ReLU()
   ▼ nn.Dropout(0.4)
   │
   ▼ nn.Linear(128, 4)    — Output layer: 4 class scores (logits)
   │
Output: (batch_size, 4)
```

**Forward pass:**
```python
def forward(self, x):
    center = x[:, x.shape[1] // 2, :]  # Take frame 15 out of 30
    return self.classifier(center)
```
It completely ignores frames 0–14 and 16–29. Only uses the center frame. This model has **NO temporal understanding** — it's equivalent to asking "what does this single image look like?" without any context of what happened before or after.

**Weight initialization:**
```python
nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2.0))
```
- **Xavier initialization**: Weights are sampled from a uniform distribution scaled by the number of inputs and outputs: `U(-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out)))`
- This ensures activations and gradients have stable variance across layers, preventing vanishing or exploding gradients at initialization
- **`gain=√2`**: Accounts for the ReLU activation (which kills half the values, so we need 2x variance to compensate)

**Result:** 89.1% accuracy, 12.7% mAP — high accuracy (because 93% is Background) but terrible event detection.

---

### MODEL B: BiLSTMClassifier (Lines 636–703) — THE BEST MODEL

**Purpose:** Captures temporal dynamics using a Bidirectional LSTM.

```python
class BiLSTMClassifier(nn.Module):
    def __init__(self, config):
        self.lstm = nn.LSTM(
            input_size=512,        # Each frame is a 512-dim vector
            hidden_size=256,       # LSTM's internal memory size
            num_layers=2,          # 2 stacked LSTM layers
            batch_first=True,      # Input shape: (batch, seq_len, features)
            bidirectional=True,    # Read forwards AND backwards
            dropout=0.4            # Dropout between layers (not within)
        )
```

**How the BiLSTM works, step by step:**

1. **Forward LSTM** reads frames 1→2→3→...→30 (left to right)
   - At each frame, it updates a 256-dim hidden state
   - Frame 15's forward hidden state contains information from frames 1–15 (the past)

2. **Backward LSTM** reads frames 30→29→28→...→1 (right to left)
   - At each frame, it updates a separate 256-dim hidden state
   - Frame 15's backward hidden state contains information from frames 30–16 (the future)

3. **Concatenation**: Frame 15's output = [forward_256 | backward_256] = 512-dim vector
   - This vector has seen ALL 30 frames — 15 frames of past AND 15 frames of future context

**Why bidirectional?** A goal celebration happens AFTER the goal. A standard LSTM reading left-to-right would see the goal first, then the celebration. But the backward LSTM sees the celebration first, then "confirms" the goal. The center frame gets the best of both worlds.

```python
lstm_output_dim = hidden_dim * 2  # 256 * 2 = 512 (bidirectional)
```

**Classifier head** (same structure as baseline):
```
BiLSTM center output (batch, 512)
   → Linear(512→256) → ReLU → Dropout → LayerNorm
   → Linear(256→128) → ReLU → Dropout
   → Linear(128→4) → Output logits
```

**Weight initialization — THE IMPORTANT DETAILS:**

```python
def _init_weights(self):
    for name, param in self.lstm.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param)    # Input-to-hidden weights
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param)         # Hidden-to-hidden weights
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)      # Zero all biases first
            n = param.size(0)
            param.data[n // 4:n // 2].fill_(1.0)  # Set FORGET GATE bias to 1
```

- **`weight_ih` (input-to-hidden)**: Xavier initialization — good for feed-forward connections
- **`weight_hh` (hidden-to-hidden)**: **Orthogonal initialization** — preserves gradient norms through time steps. Without this, multiplying the same matrix at every time step causes gradients to either explode or vanish. Orthogonal matrices have eigenvalues of magnitude 1, which keeps gradients stable.
- **Forget gate bias = 1.0**: The LSTM has 4 gates: input, forget, cell, output. The bias vector is divided into 4 equal parts. `n//4` to `n//2` is the forget gate section. Setting it to 1.0 means `sigmoid(1.0) ≈ 0.73` — the LSTM starts by **remembering 73% of information** rather than 50% (which is what sigmoid(0) gives). This is a famous trick from Jozefowicz et al. (2015) that prevents the "blank slate" problem where untrained LSTMs forget everything and can't learn long-term dependencies.

**Forward pass:**
```python
def forward(self, x):
    lstm_out, (h_n, c_n) = self.lstm(x)
    # lstm_out shape: (batch, 30, 512)  — output at every time step
    # h_n shape: (4, batch, 256)        — final hidden states (2 layers × 2 directions)
    # c_n shape: (4, batch, 256)        — final cell states
    
    center_idx = x.shape[1] // 2          # = 15
    center_output = lstm_out[:, center_idx, :]  # (batch, 512)
    return self.classifier(center_output)       # (batch, 4)
```

**Result:** 96.3% accuracy, **35.1% mAP** — dramatically better than baseline. Proves temporal modeling is essential.

---

### MODEL C: TransformerClassifier (Lines 708–792)

**Purpose:** Uses self-attention instead of recurrence. Theoretically more powerful but needs more data.

#### PositionalEncoding (Lines 708–727)
```python
pe = torch.zeros(max_len, d_model)                    # (500, 512)
position = torch.arange(0, max_len).unsqueeze(1)      # (500, 1)
div_term = torch.exp(
    torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
)
pe[:, 0::2] = torch.sin(position * div_term)  # Even indices: sine
pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices: cosine
```

**Why this exists:** Transformers process all frames simultaneously via attention. Unlike LSTMs, they have NO inherent notion of sequence order. Without positional encoding, the model cannot distinguish frame 1 from frame 30 — they're just a "bag of features." The sinusoidal encoding creates a unique "fingerprint" for each position using sine and cosine waves at different frequencies.

```python
self.register_buffer('pe', pe)
```
`register_buffer` stores the positional encoding as part of the model state (saved/loaded with the model) but it is NOT a trainable parameter (no gradients).

#### TransformerClassifier architecture:
```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=512,           # Input/output dimension
    nhead=8,               # 8 attention heads (each processes 64 dims)
    dim_feedforward=1024,  # Inner feedforward expansion
    dropout=0.4,
    batch_first=True,
    activation='gelu'      # Gaussian Error Linear Unit (smoother than ReLU)
)
self.transformer_encoder = nn.TransformerEncoder(
    encoder_layer, num_layers=2  # Stack 2 layers
)
```

Each Transformer layer does:
1. **Multi-Head Self-Attention**: Each of 30 frames "attends" to every other frame. Creates a 30×30 attention matrix. Each attention head can learn different temporal patterns (e.g., head 1 might focus on nearby frames, head 5 might focus on distant context).
2. **Feedforward Network**: Two linear layers with GELU activation: 512→1024→512
3. **Residual Connections + LayerNorm**: Adds the input back to the output (skip connection) and normalizes

**Forward pass:**
```python
def forward(self, x):
    x = self.pos_encoder(x)                            # Add positional info
    transformer_out = self.transformer_encoder(x)       # (batch, 30, 512)
    center_idx = x.shape[1] // 2
    center_output = transformer_out[:, center_idx, :]   # (batch, 512)
    return self.classifier(center_output)               # (batch, 4)
```

**Result:** 93.9% accuracy, 8.9% mAP — worse than BiLSTM. **This is a key interview talking point:**
1. **Data scale**: 60 matches (~43K sequences) is far too few for Transformers to learn meaningful attention patterns
2. **Short sequences**: Transformers shine on 1000+ token sequences where LSTMs suffer from vanishing gradients. At 30 frames, LSTM memory is more than sufficient
3. **Inductive bias**: LSTMs have a built-in bias for sequential data (process one frame at a time, accumulate state). Transformers must learn this from scratch via attention, requiring much more data.

### `get_model()` Factory Function (Lines 794–804)
```python
def get_model(model_name, config):
    models = {
        'baseline': BaselineCNN,
        'bilstm': BiLSTMClassifier,
        'transformer': TransformerClassifier,
    }
    model = models[model_name](config).to(config['device'])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```
A clean factory pattern. Creates any model by name. Also counts and prints trainable parameters.

---

## CELL 7: TRAINING ENGINE (Lines 812–1057)

The `TrainingEngine` class handles the entire training loop.

### `__init__()` (Lines 819–852)

```python
# Loss function with class weights
self.criterion = nn.CrossEntropyLoss(weight=class_weights)
```
`CrossEntropyLoss` = `LogSoftmax` + `NLLLoss`. It:
1. Converts raw logits to probabilities via softmax
2. Takes the negative log of the correct class's probability
3. Multiplies by the class weight

Example: If the model predicts `[0.1, 0.2, 0.05, 0.65]` for a Goal (class 0):
- Loss = -weight[0] × log(0.1) = -12.5 × 2.3 = **28.75** (huge penalty!)
- If it had predicted `[0.9, 0.05, 0.03, 0.02]`:
- Loss = -12.5 × log(0.9) = -12.5 × 0.105 = **1.31** (small penalty)

```python
# Adam optimizer
self.optimizer = optim.Adam(
    model.parameters(),
    lr=5e-4,              # Learning rate
    weight_decay=1e-5     # L2 regularization
)
```
**Why Adam over SGD?** Adam maintains per-parameter adaptive learning rates and momentum estimates. For our task with severe class imbalance and variable gradient magnitudes (rare events produce large gradients, background produces small gradients), Adam's adaptive rates prevent oscillation. SGD would require very careful learning rate tuning and warm-up schedules.

```python
# Learning rate scheduler
self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    self.optimizer, mode='max', factor=0.5, patience=5
)
```
Monitors validation mAP (`mode='max'` means higher is better). If mAP doesn't improve for 5 consecutive epochs (`patience=5`), it multiplies the learning rate by 0.5 (`factor=0.5`). This implements a coarse-to-fine optimization: large steps initially to find the right region, then smaller steps to fine-tune.

### `length_regularization()` (Lines 854–859)
```python
def length_regularization(self, logits):
    probs = torch.softmax(logits, dim=1)
    event_probs = 1.0 - probs[:, -1]    # P(not background)
    return torch.abs(event_probs.mean() - self.config['reg_factor'])  # |mean - 0.15|
```
**Why this exists:** Even with class weights, the model might over-predict events to avoid missing any. If it predicts 50% of frames as events (when the true rate is ~5%), this produces a large penalty: |0.50 - 0.15| = 0.35. If it predicts 15%, the penalty is zero: |0.15 - 0.15| = 0.0. This is inspired by the CA-SUM (Combining Attentive Summarization) paper which uses similar budget-aware regularization.

### `train_one_epoch()` (Lines 861–892)
```python
def train_one_epoch(self, train_loader):
    self.model.train()  # Enable dropout and batch norm training mode
    
    for batch_idx, (features, labels) in enumerate(train_loader):
        features = features.to(self.device)       # Move to GPU
        labels = labels.to(self.device).long()     # Move to GPU, ensure int64
        
        self.optimizer.zero_grad()                 # Clear previous gradients
        
        logits = self.model(features)              # Forward pass
        ce_loss = self.criterion(logits, labels)   # Cross-entropy loss
        reg_loss = self.length_regularization(logits)  # Length penalty
        loss = ce_loss + self.config['reg_lambda'] * reg_loss  # Total loss
        
        loss.backward()                            # Compute gradients (backprop)
        
        torch.nn.utils.clip_grad_norm_(            # Clip gradients
            self.model.parameters(), self.config['clip_grad']
        )
        
        self.optimizer.step()                      # Update weights
```

Step by step for each batch:
1. **`model.train()`**: Enables dropout (randomly zero neurons) and makes BatchNorm use batch statistics
2. **`optimizer.zero_grad()`**: PyTorch accumulates gradients by default. You MUST zero them before each batch or they'll add up incorrectly
3. **Forward pass**: Data flows through the model, producing logits
4. **Loss computation**: CrossEntropy + Length Regularization
5. **`loss.backward()`**: PyTorch's autograd engine walks backward through the computation graph, computing ∂loss/∂weight for every parameter
6. **Gradient clipping**: If the total gradient norm > 5.0, all gradients are rescaled proportionally. Prevents LSTM exploding gradients.
7. **`optimizer.step()`**: Adam updates each weight: `w = w - lr × adjusted_gradient`

### `evaluate()` (Lines 894–937)
```python
@torch.no_grad()
def evaluate(self, val_loader):
    self.model.eval()  # Disable dropout, use running BatchNorm stats
```
- **`@torch.no_grad()`**: Disables gradient computation. During evaluation, we don't need gradients (no backpropagation). This reduces memory by ~50% and speeds up computation by ~20%.
- **`model.eval()`**: Two critical effects:
  1. **Dropout layers stop dropping** — all neurons are active. During training, dropout randomly kills 40% of neurons. During evaluation, all neurons are used but their outputs are scaled by 0.6 (1 - dropout_rate) to compensate.
  2. **BatchNorm uses running averages** instead of batch statistics. This ensures deterministic predictions regardless of batch composition.

**mAP computation:**
```python
for c in range(self.config['num_classes'] - 1):  # Exclude Background
    binary_labels = (all_labels == c).astype(int)  # 1 if this class, 0 otherwise
    if binary_labels.sum() > 0:
        ap = average_precision_score(binary_labels, all_probs[:, c])
        aps.append(ap)
mean_ap = np.mean(aps)
```
For each event class (Goal, Cards, Substitution — NOT Background):
1. Create binary labels: 1 = this class, 0 = everything else
2. Rank all samples by the model's predicted probability for this class
3. Compute Average Precision = area under the Precision-Recall curve
4. mAP = mean of all APs

**Background is excluded** because we only care about event detection quality. High AP on Background is trivial and uninformative.

### `train()` — Full Training Loop (Lines 987–1055)
```python
for epoch in range(self.start_epoch, self.config['epochs']):
    train_loss, train_acc = self.train_one_epoch(train_loader)
    val_loss, val_acc, val_map, _, _, _ = self.evaluate(val_loader)
    
    self.scheduler.step(val_map)  # Adjust LR based on mAP
    
    is_best = val_map > self.best_val_map
    if is_best:
        self.best_val_map = val_map
        self.patience_counter = 0
    else:
        self.patience_counter += 1
    
    self.save_checkpoint(epoch, is_best=is_best)
    
    if self.patience_counter >= self.config['patience']:
        print("Early stopping...")
        break
```

Each epoch:
1. Train on all batches → get training loss and accuracy
2. Evaluate on validation set → get val loss, accuracy, and mAP
3. Feed mAP to the scheduler (may reduce LR)
4. If mAP improved → save as "best" checkpoint, reset patience
5. If mAP didn't improve → increment patience counter
6. If patience counter hits 15 → stop training (early stopping)
7. After training, load the BEST checkpoint (not the last one)

### Checkpointing (Lines 939–985)
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': self.model.state_dict(),      # All weights
    'optimizer_state_dict': self.optimizer.state_dict(), # Optimizer state (momentum, etc.)
    'scheduler_state_dict': self.scheduler.state_dict(), # LR scheduler state
    'history': self.history,                            # All loss/accuracy curves
    'best_val_map': self.best_val_map,
}
torch.save(checkpoint, path)
```
Saves EVERYTHING needed to resume training from exactly where it left off. This is critical for Colab crashes — if the session dies at epoch 15, you can resume from epoch 15, not epoch 0.

---

## CELLS 8/8a/8b/8c: TRAINING EXECUTION (Lines 1062–1191)

### `train_and_evaluate_model()` (Lines 1088–1146)
```python
def train_and_evaluate_model(model_name, ...):
    results_path = os.path.join(config['checkpoint_dir'], f'{model_name}_test_results.pth')
    best_ckpt_path = os.path.join(config['checkpoint_dir'], f'{model_name}_best.pth')
    
    # Skip if BOTH files exist (already fully trained)
    if os.path.exists(results_path) and os.path.exists(best_ckpt_path):
        print("Already trained! Loading saved results...")
        return saved_results
```
**Smart skip logic:** If both the best checkpoint AND the test results file exist, skip training entirely. This means:
- If Colab crashes DURING training → only `_latest.pth` exists → training RESUMES from last epoch
- If Colab crashes AFTER training but before next model → `_best.pth` AND `_test_results.pth` exist → SKIP

Each model is trained sequentially:
1. Cell 8a: Baseline CNN → backup downloaded to Mac
2. Cell 8b: BiLSTM → backup downloaded to Mac
3. Cell 8c: Transformer → final backup downloaded to Mac

---

## CELL 9: VISUALIZATION (Lines 1196–1335)

Generates 5 plots saved to `results/`:

1. **Training Loss Curves** (`training_loss_curves.png`): 3 subplots (one per model) showing train loss vs val loss over epochs. Look for: val loss should decrease then flatten; if it increases while train loss decreases, that's overfitting.

2. **Validation mAP Curves** (`validation_map_curves.png`): All 3 models on one plot. Shows BiLSTM clearly dominating. The BiLSTM curve rises to ~0.38 then plateaus. The Transformer barely rises above 0.1.

3. **Model Comparison Bar Chart** (`model_comparison.png`): Side-by-side bars for accuracy, mAP, and loss across all 3 models.

4. **Confusion Matrices** (`confusion_matrices.png`): 3 normalized confusion matrices. Rows = true labels, columns = predicted. Diagonal = correct predictions. Key insight: Goals and Cards are sometimes confused because they share similar visual signatures (sudden camera changes, crowd reactions).

5. **Classification Report**: Precision, recall, F1-score per class. Printed to console.

6. **Metrics JSON** (`metrics_summary.json`): Machine-readable summary of all results.

---

## CELLS 10/10b/10c: HIGHLIGHT GENERATION (Lines 1340–1737) — THE INFERENCE PIPELINE

### `knapsack_highlight_selection()` (Lines 1345–1385)

The **0/1 Knapsack Algorithm** implemented via Dynamic Programming:

```python
# Items: detected events
# Weights: clip duration (in frames)
# Values: detection confidence
# Capacity: max_duration_frames (e.g., 10 min × 60 sec × 2 fps = 1200 frames)

# DP table: K[i][w] = max value using first i items with capacity w
K = [[0.0 for _ in range(W + 1)] for _ in range(n + 1)]

for i in range(1, n + 1):
    for w in range(W + 1):
        if weights[i-1] <= w:
            # Choose the better option:
            # 1. Include item i: its value + best value with remaining capacity
            # 2. Exclude item i: same as best value without item i
            K[i][w] = max(
                values[i-1] + K[i-1][w - weights[i-1]],  # Include
                K[i-1][w]                                   # Exclude
            )
        else:
            K[i][w] = K[i-1][w]  # Can't fit, must exclude
```

**Backtracking to find selected items:**
```python
selected_indices = []
w = W
for i in range(n, 0, -1):
    if K[i][w] != K[i-1][w]:      # Item i was included
        selected_indices.insert(0, i - 1)
        w -= weights[i - 1]       # Reduce remaining capacity
```

**Why not just sort by confidence and take the top N?** Because a greedy approach doesn't consider the duration-value trade-off. Example:
- Event A: confidence 0.95, duration 5 minutes
- Event B: confidence 0.85, duration 30 seconds
- Event C: confidence 0.80, duration 30 seconds
- Budget: 6 minutes

Greedy picks: A (5 min) + B (30 sec) = 5.5 min, total value = 1.80
Knapsack picks: A (5 min) + B (30 sec) + C (30 sec) = 6 min, total value = **2.60**

### `generate_highlights()` (Lines 1387–1498)

**Step 1: Sliding window inference**
```python
all_probs = np.zeros((num_frames, num_classes))  # Accumulator for probabilities
frame_counts = np.zeros(num_frames)               # Count of predictions per frame

for start in range(0, num_frames - seq_len + 1):
    seq = torch.tensor(features[start:start + 30]).unsqueeze(0).to(device)
    logits = model(seq)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    center = start + seq_len // 2    # Center frame index
    all_probs[center] += probs       # Accumulate probabilities
    frame_counts[center] += 1        # Track how many predictions
```
Unlike training (which uses stride=15), inference uses stride=1. Every possible 30-frame window is evaluated. Each frame gets predictions from up to 30 overlapping windows.

```python
valid = frame_counts > 0
all_probs[valid] /= frame_counts[valid, np.newaxis]  # Average overlapping predictions
```
Averaging multiple predictions for each frame produces smoother, more reliable output than a single prediction.

**Step 2: Thresholding**
```python
bg_class = config['num_classes'] - 1    # = 3 (Background)
event_mask = all_probs[:, :bg_class].max(axis=1) > 0.3
```
A frame is marked as an "event" if ANY non-background class probability exceeds 0.3 (30%). We use 0.3 instead of the default 0.5 to improve recall — the model is often less confident on custom videos than on SoccerNet data, so a lower threshold catches more events.

**Step 3: Group consecutive event frames**
```python
for i in range(num_frames):
    if event_mask[i] and not in_event:
        event_start = i           # Start of a new event
        in_event = True
    elif not event_mask[i] and in_event:
        # End of event — create event record
        best_class = all_probs[event_start:i, :bg_class].mean(axis=0).argmax()
        confidence = float(all_probs[event_start:i, best_class].mean())
```
If frames 100–120 are all flagged as events, they become ONE event. The event's class is determined by the average probability across all flagged frames, and the confidence is the average probability of the winning class.

**Step 4: Add padding**
```python
pad = 30 * config['target_fps']  # 30 seconds × 2 FPS = 60 frames
events.append({
    'start': max(0, event_start - pad),   # 30 seconds BEFORE the event
    'end': min(num_frames, i + pad),       # 30 seconds AFTER the event
    ...
})
```
Each event clip includes ±30 seconds of context. A goal at 23:45 becomes a clip from 23:15 to 24:15, showing the build-up play, the goal itself, and the celebration.

**Step 5: Knapsack selection**
```python
max_frames = max_highlight_minutes * 60 * config['target_fps']  # 10 × 60 × 2 = 1200
selected_events = knapsack_highlight_selection(all_events, max_frames)
selected_events.sort(key=lambda e: (e['half'], e['start']))  # Sort chronologically
```

### `extract_features_from_video()` (Lines 1584–1665)

This is the function that handles CUSTOM videos (not SoccerNet data).

```python
# Load ResNet-152 (matching SoccerNet's feature extractor)
resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
modules = list(resnet.children())[:-1]     # Remove the final classification layer
feature_extractor = nn.Sequential(*modules).to(device)
feature_extractor.eval()
```
- **`models.resnet152(weights=...DEFAULT)`**: Loads ResNet-152 pre-trained on ImageNet (1000 classes, 1.2M images). The weights are frozen (not trained further).
- **`list(resnet.children())[:-1]`**: ResNet-152 has: [Conv1, BN1, ReLU, MaxPool, Layer1, Layer2, Layer3, Layer4, AvgPool, **FC(2048→1000)**]. We remove the last FC layer, so the output is the 2048-dim feature vector before classification.

```python
preprocess = transforms.Compose([
    transforms.Resize(256),           # Resize shorter edge to 256px
    transforms.CenterCrop(224),       # Crop center 224×224 (ImageNet standard)
    transforms.ToTensor(),            # Convert PIL image to tensor [0,1]
    transforms.Normalize(             # ImageNet normalization values
        mean=[0.485, 0.456, 0.406],   # Subtract mean
        std=[0.229, 0.224, 0.225]     # Divide by std
    ),
])
```
These EXACT normalization values are required because ResNet-152 was trained with them. Using different values would shift the input distribution and degrade performance.

```python
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)          # e.g., 30 FPS for most videos
frame_interval = max(1, int(round(fps / target_fps)))  # 30/2 = 15 → process every 15th frame
```
If the video is 30 FPS and we want 2 FPS, we process every 15th frame.

```python
# PCA: 2048 → 512 dimensions
pca = PCA(n_components=512)
features_512 = pca.fit_transform(raw_features)
```
**Why PCA?** SoccerNet's pre-computed features are 512-dim (ResNet-152 2048-dim reduced by PCA). To ensure our custom video features live in the same mathematical space, we also apply PCA 2048→512. This resolves the **domain shift** problem.

### `create_highlight_video()` (Lines 1676–1737)

```python
from moviepy.editor import VideoFileClip, concatenate_videoclips

video = VideoFileClip(video_path)
clips = []

for evt in events:
    start_sec = max(0, evt['start'] / target_fps)     # Frame index → seconds
    end_sec = min(video.duration, evt['end'] / target_fps)
    clip = video.subclip(start_sec, end_sec)           # Cut the clip
    clips.append(clip)

final_video = concatenate_videoclips(clips)            # Join all clips
final_video.write_videofile(
    output_path,
    codec="libx264",     # H.264 — universally supported video codec
    audio_codec="aac",   # AAC — standard audio companion to H.264
    logger=None          # Suppress MoviePy's verbose progress output
)
```

---

## THE STREAMLIT APP (app.py) <a name="section-4"></a>

[app.py](file:///Users/vaibhav/Study/Sem%206/Deep%20Learning/Project/app.py) is a 656-line Streamlit web dashboard that provides a visual interface for running inference.

### Key Components:

1. **Model Definition (Lines 168–198)**: A copy of `BiLSTMClassifier` (must match training code exactly or weights won't load)

2. **`@st.cache_resource` Model Loading (Lines 353–363)**: Loads the model ONCE and caches it. Subsequent runs don't re-load.

3. **Sidebar Controls (Lines 378–431)**: Detection sensitivity slider (0.1–0.9), max highlight duration slider (1–15 min)

4. **File Uploads (Lines 438–455)**: User uploads `.npy` features file + original `.mp4` video

5. **Processing Pipeline (Lines 476–643)**: Runs `generate_highlights()` → `stitch_highlight_video()` → displays video player, timeline chart, and detection details table

---

# 5. KEY DEEP LEARNING CONCEPTS <a name="section-5"></a>

### What is mAP (Mean Average Precision)?
Imagine you have 1000 frames and 10 of them are goals. Your model ranks all 1000 frames by "goal probability."
- If the top 10 ranked frames are ALL actual goals → AP = 1.0 (perfect)
- If actual goals are at ranks 1, 10, 50, 80, 200, 300, 500, 700, 800, 999 → AP ≈ 0.3 (poor)
- mAP = average of AP across Goal, Cards, and Substitution classes
- **Background is excluded** because we care about event detection, not background detection

### What is Domain Shift?
Training data came from ResNet-152. Initial inference used ResNet-18. Even though both produce 512-dim vectors, the mathematical meaning of each dimension is different. It's like two people speaking different languages — both use words, but the words mean different things.

**Analogy:** You learned to recognize dogs from photographs. Now I show you watercolor paintings of dogs. Same dogs, completely different visual style. You struggle because your brain learned "photo features" not "painting features."

**Solution:** Use the SAME ResNet-152 for both training and inference. Now both speak the same "language."

### Why BiLSTM beats Transformer?
1. **Inductive bias**: LSTMs assume sequential processing (frame by frame). Transformers assume nothing — they must learn temporal patterns from data via attention.
2. **Data efficiency**: 60 matches ≈ 43K training sequences. Transformers typically need 100K–1M+ samples to outperform LSTMs.
3. **Sequence length**: Transformers excel on 1000+ token sequences where LSTMs forget early context. At 30 frames, LSTM memory is fully sufficient.

### What is Gradient Clipping?
LSTMs propagate gradients backward through time. At each time step, the gradient is multiplied by the weight matrix. After 30 time steps, this is W³⁰. If the largest eigenvalue of W is >1, the gradient explodes to infinity (NaN). Gradient clipping caps the total gradient norm at 5.0, preventing this.

### What is Early Stopping?
If validation mAP stops improving, continuing to train only leads to overfitting (memorizing training data). Early stopping halts training and uses the best-performing checkpoint. Our patience is 15 epochs — it waits 15 epochs for improvement before stopping.

---

# 6. 40+ INTERVIEW QUESTIONS WITH MODEL ANSWERS <a name="section-6"></a>

## Category A: Architecture & Design

**Q1: Why a two-stage pipeline instead of end-to-end?**
> End-to-end training requires holding both the CNN and temporal model in GPU memory simultaneously with all intermediate activations for backpropagation. This needs 40+ GB of VRAM. Colab's free T4 has only 15GB. The two-stage approach decouples spatial feature extraction (done once, saved as .npy files) from temporal modeling (trained iteratively), reducing memory to ~2GB during training.

**Q2: Why ResNet-152 and not a smaller model like ResNet-18 or ResNet-50?**
> The SoccerNet dataset provides pre-extracted features using ResNet-152 + PCA. Our temporal model was trained on these features. When we initially used ResNet-18 for custom video inference, the feature distributions were incompatible — the feature vectors lived in different mathematical spaces. This is called "domain shift." We resolved it by using ResNet-152 for inference too, ensuring the features are in the same space.

**Q3: What is PCA and why reduce from 2048 to 512?**
> PCA (Principal Component Analysis) is a dimensionality reduction technique that finds the directions of maximum variance in the data. It projects 2048 dimensions down to the 512 most informative dimensions. This reduces memory, speeds up training, and removes noise from less important dimensions. We use 512 specifically because that's what SoccerNet uses — our features must match.

**Q4: Why BiLSTM over a standard (unidirectional) LSTM?**
> A standard LSTM reads frames left-to-right and only uses past context. For event detection, future context is critical. A crowd celebration 3 seconds AFTER a goal helps confirm the goal. The BiLSTM reads both directions: the forward pass captures past context, the backward pass captures future context. The center frame gets the best of both worlds.

**Q5: Why did the Transformer underperform?**
> Three reasons: (1) Data scale — Transformers need massive datasets to learn attention patterns; we only had 60 matches. (2) Sequence length — Transformers shine on 1000+ token sequences; our windows are only 30 frames. (3) Inductive bias — LSTMs have a built-in assumption that data is sequential; Transformers must learn this from scratch, needing more data.

**Q6: Why use the center frame's output instead of the last frame's output?**
> In a BiLSTM, the center frame has seen equal context from both directions — 15 frames of past and 15 frames of future. The last frame has seen all past context but minimal future context. Since we want balanced temporal context for classification, the center frame is optimal.

**Q7: Explain the Knapsack algorithm and why you used it.**
> The 0/1 Knapsack is a Dynamic Programming algorithm that solves: "Given N items with weights and values, and a capacity limit, maximize total value." In our case: items = detected events, weight = clip duration, value = detection confidence, capacity = 10-minute highlight budget. A greedy approach (pick top-N by confidence) doesn't consider the duration-value trade-off. Knapsack guarantees the mathematically optimal selection.

**Q8: Why 4 classes and not all 17 SoccerNet classes?**
> With only 60 matches, most of the 17 original SoccerNet classes have fewer than 10 examples each. Training on such sparse data leads to severe overfitting. By merging into 4 classes (Goal, Cards, Substitution, Background), each class has hundreds of examples, making training statistically robust.

## Category B: Training & Optimization

**Q9: How did you handle the class imbalance problem?**
> 95-99% of frames are Background. Without handling, the model would predict "Background" for everything and achieve 95%+ accuracy while detecting zero events. Solution: inverse frequency class weighting in CrossEntropyLoss. Background weight ≈ 0.26, Goal weight ≈ 12.5. This makes the loss function penalize event misclassifications 50x more than background misclassifications.

**Q10: What is length regularization?**
> A penalty term: |mean(event_probability) - 0.15|. It targets a 15% event rate across the batch. Without it, the model might over-predict events (better safe than sorry). With it, the model is penalized if the average event prediction rate deviates from 15%. Inspired by the CA-SUM paper.

**Q11: Why Adam optimizer instead of SGD?**
> Adam maintains per-parameter adaptive learning rates and momentum estimates. With severe class imbalance, rare events produce large gradients while background produces small gradients. Adam handles this variability automatically. SGD with a fixed learning rate would oscillate or diverge without extensive hyperparameter tuning.

**Q12: Explain gradient clipping.**
> LSTMs propagate gradients backward through 30 time steps. At each step, the gradient is multiplied by the weight matrix. If the largest eigenvalue > 1, gradients grow exponentially: W³⁰ → infinity. clip_grad_norm_(params, 5.0) rescales all gradients proportionally if their combined L2 norm exceeds 5.0. This prevents NaN losses and training instability.

**Q13: What is ReduceLROnPlateau?**
> A learning rate scheduler that monitors validation mAP. If mAP doesn't improve for 5 consecutive epochs (patience=5), it multiplies the learning rate by 0.5 (factor=0.5). This implements coarse-to-fine optimization: large steps initially to find the right region, then small steps to fine-tune. In practice, the LR drops from 5e-4 → 2.5e-4 → 1.25e-4 over 23 epochs.

**Q14: Why Xavier initialization?**
> Xavier sets weights from a distribution scaled by fan-in and fan-out: W ~ U(-√(6/(n_in+n_out)), √(6/(n_in+n_out))). This ensures activations and gradients have stable variance across layers, preventing vanishing or exploding gradients at initialization. The gain=√2 compensates for ReLU killing half the values.

**Q15: What is the forget gate bias trick?**
> The LSTM bias vector is divided into 4 equal parts: [input gate | forget gate | cell gate | output gate]. Setting the forget gate bias to 1.0 gives sigmoid(1.0) ≈ 0.73, meaning the LSTM starts by retaining 73% of information. Without this, sigmoid(0) = 0.5, and the LSTM forgets half of everything immediately, making it unable to learn long-term dependencies early in training.

**Q16: Why LayerNorm instead of BatchNorm?**
> LayerNorm normalizes across features within a single sample (independent of batch size). BatchNorm normalizes across the batch dimension and requires different behavior during training vs inference (running statistics). LayerNorm works identically in both modes, is more stable with small batch sizes, and is the standard choice for sequence models.

**Q17: Why drop_last=True in the training DataLoader?**
> If the last batch has fewer than 32 samples (e.g., 7), BatchNorm statistics become unreliable and the length regularization term becomes noisy. drop_last=True discards this incomplete batch. We lose ~30 samples out of ~43,000 — negligible.

**Q18: Why orthogonal initialization for LSTM hidden-to-hidden weights?**
> The hidden-to-hidden weight matrix is applied repeatedly at every time step: h_t = f(W_ih * x_t + W_hh * h_{t-1}). Orthogonal matrices preserve the L2 norm of vectors they multiply: ||Wx|| = ||x||. This means gradients neither grow nor shrink through time steps, maintaining stable training dynamics.

## Category C: Evaluation & Results

**Q19: What does 35.1% mAP actually mean?**
> mAP measures ranking quality. For each event class, we rank all frames by predicted probability and compute Average Precision (area under precision-recall curve). 35.1% means our model correctly identifies about 1 in 3 event frames above background frames. SoccerNet SOTA is ~70% with 500 matches. We achieve 50% of SOTA with 12% of the data — a strong result proving the architecture is sound and data is the bottleneck.

**Q20: Why is test accuracy 96% but mAP only 35%?**
> Class imbalance. If 95% of frames are Background, predicting "Background" for everything gives 95% accuracy. High accuracy is a misleading metric for rare-event detection. mAP directly measures event detection quality, ignoring Background entirely. 96% accuracy with 35% mAP means the model is excellent at Background but still misses some events.

**Q21: What's in the confusion matrix?**
> Rows = true labels, columns = predicted labels. Diagonal = correct predictions. Off-diagonal = errors. Our BiLSTM's confusion matrix shows Goals are sometimes confused with Cards (both involve sudden camera changes, crowd reactions, player clustering). Background is predicted with 97%+ accuracy.

**Q22: What is the precision-recall trade-off?**
> Precision = true positives / (true positives + false positives). "Of what I predicted as goals, how many were actually goals?" Recall = true positives / (true positives + false negatives). "Of all actual goals, how many did I catch?" Lowering the detection threshold increases recall (catch more events) but decreases precision (more false alarms). Our threshold of 0.3 favors recall.

## Category D: Data & Pipeline

**Q23: What are the .npy files?**
> NumPy array files containing pre-extracted features. Each file stores a 2D array of shape (num_frames, 512). Each row is a 512-dimensional feature vector representing one video frame at 2 FPS. These were extracted by the SoccerNet team using ResNet-152 (TF2) + PCA.

**Q24: Explain the sliding window approach.**
> We take 30 consecutive frames (15 seconds at 2 FPS). The model predicts the class of the CENTER frame (frame 15). Then we slide the window forward. During training, we slide by 15 frames (50% overlap). During inference, we slide by 1 frame (maximum overlap). Multiple overlapping predictions for each frame are averaged for smoother output.

**Q25: How does event detection work after model inference?**
> The model produces per-frame class probabilities. We threshold at 0.3 — any frame where a non-background class exceeds 30% is marked as an "event frame." Consecutive event frames are grouped into one event. Each event gets ±30 seconds of padding. Finally, the Knapsack algorithm selects the optimal combination under the time budget.

**Q26: Why use the SoccerNet "test" split for training?**
> SoccerNet's train split (300 matches) is too large for Colab's 112GB disk. The test split (~100 matches) is the smallest. We download it and re-split internally into 60/20/20. This means we're not using SoccerNet's official splits, but our internal split is valid as long as there's no data leakage between our train/val/test sets.

## Category E: Domain Shift & Feature Extraction

**Q27: Explain the domain shift problem in full detail.**
> Our BiLSTM was trained on feature vectors from ResNet-152 + PCA (SoccerNet's pipeline). In our initial version, we extracted features from custom YouTube videos using ResNet-18. Although both produced 512-dim vectors, the features were fundamentally different. ResNet-18 has 11M parameters and 4 residual groups. ResNet-152 has 60M parameters and much deeper residual groups. The same image produces completely different activation patterns. The BiLSTM received out-of-distribution inputs and couldn't classify events.

**Q28: How did you fix domain shift?**
> We upgraded the custom video pipeline to use the exact same ResNet-152 architecture + PCA reduction (2048→512). Now both the training data and custom video features are extracted by the same model, living in the same mathematical space.

**Q29: What other approaches could fix domain shift?**
> (1) Learned projection layer: Train a small neural network to map ResNet-18 features into the ResNet-152 space. Requires paired features from both models on the same frames. (2) End-to-end fine-tuning: Train CNN + LSTM jointly on raw video frames. Eliminates the feature extraction dependency but requires multi-GPU hardware. (3) Domain adaptation techniques like CORAL or MMD loss.

## Category F: Code-Specific Questions

**Q30: What does model.eval() do?**
> Switches from training to evaluation mode. (1) Dropout layers stop randomly zeroing neurons — all neurons are active, but scaled by (1-p). (2) BatchNorm layers use accumulated running mean/variance instead of batch statistics. Without model.eval(), predictions are non-deterministic and degraded.

**Q31: What does @torch.no_grad() do?**
> Disables gradient computation for the decorated function. During inference, we don't need gradients. Disabling them reduces memory by ~50% (no computation graph stored) and speeds up computation by ~20%.

**Q32: What does optimizer.zero_grad() do?**
> PyTorch accumulates gradients by default (adds new gradients to existing ones). Without zeroing, gradients from the previous batch would be added to the current batch's gradients, producing incorrect weight updates. This must be called at the start of every training iteration.

**Q33: What does .unsqueeze(0) do?**
> Adds a batch dimension. A single sequence has shape (30, 512). The model expects (batch_size, 30, 512). unsqueeze(0) converts (30, 512) → (1, 30, 512) — a batch of size 1.

**Q34: What codec is used for the output video?**
> H.264 (libx264) with AAC audio. H.264 is the most universally supported video codec — playable on every device, browser, and media player. AAC is the standard audio companion.

**Q35: What does register_buffer do in PositionalEncoding?**
> Stores the positional encoding as part of the model state (saved/loaded with checkpoints, moved to GPU with model.to(device)) but it is NOT a trainable parameter — no gradients are computed for it.

## Category G: Future Work & Comparisons

**Q36: What would you do with unlimited compute?**
> (1) Train on all 500 SoccerNet matches — would likely push mAP to 50%+. (2) Add audio features — crowd noise spikes strongly correlate with goals. (3) Implement multi-scale temporal windows — 5s for cards, 30s for goal sequences. (4) Use a larger Transformer with pre-training on a large video corpus. (5) Add YOLO+DeepSORT tracking for player-level analytics.

**Q37: How does this differ from commercial systems?**
> Commercial systems like WSC Sports and IBM Watson Media use 10,000+ matches, multi-modal inputs (video + audio + text overlays + player tracking), human-in-the-loop refinement, and inference farms of GPUs. Our system demonstrates the same core algorithmic pipeline at academic scale. The architecture is sound; the performance gap is purely a function of data and compute.

**Q38: Why not use audio features?**
> We didn't because SoccerNet's pre-extracted features are video-only and we wanted to match their pipeline exactly. However, audio is an incredibly powerful signal — crowd noise correlates strongly with goals and is architecture-agnostic. This is one of the top improvements for future work.

**Q39: What is the most interesting thing you learned?**
> "That the Transformer is not always better than the LSTM. In the literature, Transformers dominate NLP benchmarks, but our experiments show that with limited data, the LSTM's sequential inductive bias gives it a 4x mAP advantage. This taught me that architecture selection should be driven by data characteristics, not by popularity."

**Q40: Why did you build a Streamlit app?**
> The Streamlit app (app.py) provides a professional web interface for demonstration. It loads the trained BiLSTM model, allows users to upload pre-extracted features and the source video, runs inference with adjustable sensitivity, and produces a downloadable highlight video with a visual timeline. This makes the system accessible to non-technical users.

**Q41: What is the time complexity of your inference pipeline?**
> Feature extraction: O(N) where N = number of frames. Each frame goes through ResNet-152 once. Temporal inference: O(N × S × H) where S = sequence length, H = hidden dimension. Knapsack: O(E × W) where E = number of events, W = max frames. Video stitching: O(D) where D = total clip duration. Total: dominated by feature extraction (ResNet-152 is expensive).

**Q42: What if someone asks about a line you don't recognize?**
> Stay calm. Say: "That's a utility function / preprocessing step. Let me walk you through the logic..." Then explain what you think it does based on variable names and context. Most interviewers care about your reasoning ability, not memorization.
