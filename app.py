"""
🏟️ Football Highlight Detection — Streamlit Dashboard
======================================================
A professional web interface for the Football Highlight Detection pipeline.
Runs entirely on your local machine (Mac CPU). No Colab needed.

Usage:
    streamlit run app.py
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import tempfile
import time
import cv2
from PIL import Image
from torchvision import models, transforms
from sklearn.decomposition import PCA

# ============================================================
# 1. PAGE CONFIG & STYLING
# ============================================================
st.set_page_config(
    page_title="MatchCut AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #a0a0cc;
        font-size: 1rem;
        margin: 0;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-card h3 {
        color: #8b8bba;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    .metric-card .value {
        color: #e0e0ff;
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* Status badges */
    .status-ready {
        background: #1a3a2a;
        color: #4ade80;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #2d5a3d;
    }
    .status-waiting {
        background: #3a2a1a;
        color: #fbbf24;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #5a4a2d;
    }

    /* Pipeline step */
    .pipeline-step {
        background: #1a1a2e;
        border-left: 4px solid #6366f1;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.8rem;
        color: #c0c0e0;
    }
    .pipeline-step.active {
        border-left-color: #22c55e;
        background: #1a2a1e;
    }
    .pipeline-step.done {
        border-left-color: #22c55e;
        color: #4ade80;
    }

    /* Timeline bar */
    .timeline-container {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #2a2a4a;
    }

    /* Hide default streamlit stuff */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #c0c0e0;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 2. MODEL DEFINITION (Must match training code exactly)
# ============================================================
class BiLSTMClassifier(nn.Module):
    """CNN features → Bidirectional LSTM → Event Classification."""
    def __init__(self, feature_dim=512, hidden_dim=256, num_layers=2,
                 num_classes=4, dropout=0.4):
        super().__init__()
        self.name = "CNN_BiLSTM"
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        lstm_output_dim = hidden_dim * 2
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

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        center_idx = x.shape[1] // 2
        center_output = lstm_out[:, center_idx, :]
        return self.classifier(center_output)


# ============================================================
# 3. FEATURE EXTRACTION & INFERENCE PIPELINE
# ============================================================
@st.cache_data(show_spinner=False)
def extract_features_from_video(video_path, target_fps=2, batch_size=16):
    """
    Extracts ResNet-152 features from video with batched processing for speed,
    then applies PCA to reduce from 2048-dim to 512-dim.
    """
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    
    
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
        return None

    frame_interval = max(1, int(round(fps / target_fps)))
    raw_features = []
    
    count = 0
    batch_frames = []
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                input_tensor = preprocess(img)
                batch_frames.append(input_tensor)
                
                if len(batch_frames) == batch_size:
                    batch_tensor = torch.stack(batch_frames).to(device)
                    feat = feature_extractor(batch_tensor)
                    feat = feat.view(batch_size, -1).cpu().numpy()
                    raw_features.extend(feat)
                    batch_frames = []
                    
            count += 1
            
        if len(batch_frames) > 0:
            batch_tensor = torch.stack(batch_frames).to(device)
            feat = feature_extractor(batch_tensor)
            feat = feat.view(len(batch_frames), -1).cpu().numpy()
            raw_features.extend(feat)

    cap.release()
    raw_features = np.array(raw_features)
    
    if raw_features.shape[0] < 512:
        pca = PCA(n_components=raw_features.shape[0])
        features_reduced = pca.fit_transform(raw_features)
        features_512 = np.pad(features_reduced, ((0,0), (0, 512 - raw_features.shape[0])))
    else:
        pca = PCA(n_components=512)
        features_512 = pca.fit_transform(raw_features)
        
    return features_512.astype(np.float32)


def knapsack_highlight_selection(events, max_duration_frames):
    """0/1 Knapsack DP to select optimal highlight clips under a time budget."""
    if not events:
        return []
    n = len(events)
    weights = [e['end'] - e['start'] for e in events]
    values = [e['confidence'] for e in events]
    W = max_duration_frames

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

    selected_indices = []
    w = W
    for i in range(n, 0, -1):
        if K[i][w] != K[i-1][w]:
            selected_indices.insert(0, i - 1)
            w -= weights[i - 1]
    return [events[i] for i in selected_indices]


@torch.no_grad()
def generate_highlights(model, features_path, threshold, max_highlight_minutes,
                        seq_len=30, num_classes=4, target_fps=2, progress_cb=None):
    """Run BiLSTM inference on pre-extracted features."""
    model.eval()
    device = next(model.parameters()).device

    features = np.load(features_path)
    num_frames = features.shape[0]

    all_probs = np.zeros((num_frames, num_classes))
    frame_counts = np.zeros(num_frames)

    total_windows = max(1, num_frames - seq_len + 1)

    for i, start in enumerate(range(0, num_frames - seq_len + 1)):
        seq = torch.tensor(
            features[start:start + seq_len], dtype=torch.float32
        ).unsqueeze(0).to(device)

        logits = model(seq)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        center = start + seq_len // 2
        all_probs[center] += probs
        frame_counts[center] += 1

        if progress_cb and i % 200 == 0:
            progress_cb(i / total_windows)

    valid = frame_counts > 0
    all_probs[valid] /= frame_counts[valid, np.newaxis]

    # Detect events (threshold on non-background probability)
    bg_class = num_classes - 1
    event_mask = all_probs[:, :bg_class].max(axis=1) > threshold

    # Group consecutive detections
    events = []
    in_event = False
    event_start = 0

    for i in range(num_frames):
        if event_mask[i] and not in_event:
            event_start = i
            in_event = True
        elif not event_mask[i] and in_event:
            confidence = float(all_probs[event_start:i, :bg_class].max(axis=1).mean())
            pad = 30 * target_fps  # ±30 seconds
            events.append({
                'start': max(0, event_start - pad),
                'end': min(num_frames, i + pad),
                'event_frame': (event_start + i) // 2,
                'confidence': confidence,
                'time_seconds': event_start / target_fps,
            })
            in_event = False

    if in_event:
        i = num_frames
        confidence = float(all_probs[event_start:i, :bg_class].max(axis=1).mean())
        pad = 30 * target_fps
        events.append({
            'start': max(0, event_start - pad),
            'end': min(num_frames, i + pad),
            'event_frame': (event_start + i) // 2,
            'confidence': confidence,
            'time_seconds': event_start / target_fps,
        })

    # Knapsack selection
    max_frames = max_highlight_minutes * 60 * target_fps
    selected = knapsack_highlight_selection(events, max_frames)
    selected.sort(key=lambda e: e['start'])

    return selected, events


def stitch_highlight_video(video_path, events, output_path, target_fps=2,
                           progress_cb=None):
    """Cut and merge highlight clips from the source video."""
    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips
    except ImportError:
        from moviepy import VideoFileClip, concatenate_videoclips

    video = VideoFileClip(video_path)
    clips = []

    for i, evt in enumerate(events):
        start_sec = max(0, evt['start'] / target_fps)
        end_sec = min(video.duration, evt['end'] / target_fps)
        if end_sec <= start_sec:
            continue
        if hasattr(video, 'subclip'):
            clip = video.subclip(start_sec, end_sec)
        else:
            clip = video.subclipped(start_sec, end_sec)
        clips.append(clip)
        if progress_cb:
            progress_cb((i + 1) / len(events))

    if not clips:
        return False

    final_video = concatenate_videoclips(clips)
    final_video.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        logger=None,
    )
    video.close()
    final_video.close()
    return True


# ============================================================
# 4. LOAD MODEL (cached so it only loads once)
# ============================================================
@st.cache_resource
def load_model(checkpoint_path):
    """Load the trained BiLSTM model from a .pth checkpoint."""
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    model = BiLSTMClassifier()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    best_map = checkpoint.get('best_val_map', 'N/A')
    epoch = checkpoint.get('epoch', 'N/A')
    return model, best_map, epoch


# ============================================================
# 5. STREAMLIT UI
# ============================================================

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>⚽ MatchCut AI</h1>
    <p>AI-Powered Highlight Generation using CNN + Bi-LSTM</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # Model checkpoint
    st.markdown("### 🧠 Model")
    ckpt_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'CNN_BiLSTM_best.pth')
    model_loaded = os.path.exists(ckpt_path)

    if model_loaded:
        st.markdown('<span class="status-ready">✓ Model Found</span>', unsafe_allow_html=True)
        model, best_map, best_epoch = load_model(ckpt_path)
        if isinstance(best_map, float):
            st.caption(f"Best mAP: {best_map:.4f} | Epoch: {best_epoch}")
    else:
        st.markdown('<span class="status-waiting">⚠ Model Not Found</span>', unsafe_allow_html=True)
        st.caption(f"Expected at: `checkpoints/CNN_BiLSTM_best.pth`")
        model = None

    st.markdown("---")

    # Controls
    st.markdown("### 🎚️ Controls")
    threshold = st.slider(
        "Detection Sensitivity",
        min_value=0.10, max_value=0.90, value=0.30, step=0.05,
        help="Lower = more highlights detected (but less confident). Higher = fewer but more confident highlights."
    )
    max_duration = st.slider(
        "Max Highlight Duration (minutes)",
        min_value=1, max_value=15, value=7, step=1,
        help="Maximum total length of the final highlight reel."
    )

    st.markdown("---")

    # Architecture info
    st.markdown("### 🏗️ Architecture")
    st.markdown("""
    ```
    Raw Video
      ↓
    ResNet-152 (Feature Extraction)
      ↓
    512-dim Feature Vectors
      ↓
    Bi-LSTM (Temporal Modeling)
      ↓
    Event Detection + Knapsack
      ↓
    Highlight Video
    ```
    """)

# --- Main Content ---
if not model_loaded:
    st.error("❌ Model checkpoint not found. Place `CNN_BiLSTM_best.pth` in the `checkpoints/` folder.")
    st.stop()

# File uploads
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎬 Source Video")
    video_file = st.file_uploader(
        "Upload the original `.mp4` match video",
        type=["mp4", "avi", "mkv"],
        help="Required to generate the final highlight video."
    )

with col2:
    st.markdown("### ⚡ Pre-Extracted Features (Optional)")
    features_file = st.file_uploader(
        "Upload `.npy` features for instant inference",
        type=["npy"],
        help="Skips the heavy ResNet-152 extraction. Perfect for live demos."
    )

# Status indicators
st.markdown("---")
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    if video_file:
        st.markdown('<span class="status-ready">✓ Video Loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-waiting">⏳ Awaiting Video</span>', unsafe_allow_html=True)
with col_s2:
    if features_file:
        st.markdown('<span class="status-ready">✓ Fast Mode (NPY)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-waiting">⏳ Full Extraction Mode</span>', unsafe_allow_html=True)
with col_s3:
    st.markdown('<span class="status-ready">✓ AI Models Ready</span>', unsafe_allow_html=True)

st.markdown("---")

# Generate button
if video_file:
    if st.button("🚀 Generate Highlights", use_container_width=True):

        # Save uploaded files to temp directory
        tmp_dir = tempfile.mkdtemp()
        video_path = os.path.join(tmp_dir, "source_video.mp4")
        features_path = os.path.join(tmp_dir, "features.npy")
        output_path = os.path.join(tmp_dir, "highlights.mp4")

        with open(video_path, 'wb') as f:
            f.write(video_file.getvalue())

        st.markdown("### 🔄 Processing Pipeline")
        
        extract_time = 0
        
        if features_file:
            # Fast Path: User provided features
            progress_bar = st.progress(0, text="Step 1/3: Loading pre-extracted features (Lightning Fast!)...")
            with open(features_path, 'wb') as f:
                f.write(features_file.getvalue())
            features_data = np.load(features_path)
            progress_bar.progress(0.4, text="Step 1/3: Features loaded instantly.")
        else:
            # Full Path: Extract features using ResNet-152
            progress_bar = st.progress(0, text="Step 1/3: Extracting visual features using ResNet-152...")
            extract_start = time.time()
            with st.spinner("Extracting features with ResNet-152 (Batched for speed)... This may take a moment."):
                features_data = extract_features_from_video(video_path, batch_size=16)
                if features_data is None:
                    st.error("Failed to extract features from video.")
                    st.stop()
                np.save(features_path, features_data)
            extract_time = time.time() - extract_start
            progress_bar.progress(0.4, text=f"Step 1/3: Features extracted in {extract_time:.1f}s")

        num_frames = features_data.shape[0]
        video_duration_min = num_frames / 2 / 60  # 2 FPS

        # Metrics row
        st.markdown("### 📊 Input Summary")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Frames", f"{num_frames:,}")
        with m2:
            st.metric("Video Duration", f"{video_duration_min:.1f} min")
        with m3:
            st.metric("Feature Dim", f"{features_data.shape[1]}")
        with m4:
            st.metric("Target FPS", "2")

        # --- STEP 2: BiLSTM Inference ---
        def update_inference_progress(pct):
            progress_bar.progress(min(0.4 + pct * 0.3, 0.7),
                                  text=f"Step 2/3: Bi-LSTM inference... {pct*100:.0f}%")

        start_time = time.time()
        selected_events, all_events = generate_highlights(
            model, features_path, threshold, max_duration,
            progress_cb=update_inference_progress
        )
        inference_time = time.time() - start_time

        progress_bar.progress(0.7, text=f"Step 2/3: Detection complete! ({inference_time:.1f}s)")

        # --- STEP 3: Video Stitching ---
        if selected_events:
            progress_bar.progress(0.75, text="Step 3/3: Stitching highlight video...")

            def update_stitch_progress(pct):
                progress_bar.progress(min(0.75 + pct * 0.25, 1.0),
                                      text=f"Step 3/3: Stitching clips... {pct*100:.0f}%")

            stitch_start = time.time()
            success = stitch_highlight_video(
                video_path, selected_events, output_path,
                progress_cb=update_stitch_progress
            )
            stitch_time = time.time() - stitch_start
            total_time = extract_time + inference_time + stitch_time

            progress_bar.progress(1.0, text=f"✅ Complete! Total time: {total_time:.1f}s")

            if success:
                st.balloons()

                # Results section
                st.markdown("---")
                st.markdown("### 🎬 Generated Highlights")

                # Result metrics
                r1, r2, r3, r4 = st.columns(4)
                total_highlight_sec = sum(
                    (e['end'] - e['start']) / 2 for e in selected_events
                )
                with r1:
                    st.metric("Highlights Found", f"{len(selected_events)}")
                with r2:
                    st.metric("Highlight Duration", f"{total_highlight_sec/60:.1f} min")
                with r3:
                    avg_conf = np.mean([e['confidence'] for e in selected_events])
                    st.metric("Avg Confidence", f"{avg_conf:.1%}")
                with r4:
                    st.metric("Processing Time", f"{total_time:.1f}s")

                # Video player
                st.markdown("#### ▶️ Watch Your Highlights")
                with open(output_path, 'rb') as vf:
                    video_bytes = vf.read()
                st.video(video_bytes)

                # Download button
                st.download_button(
                    label="⬇️ Download Highlight Video",
                    data=video_bytes,
                    file_name="highlights.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )

                # Event timeline
                st.markdown("#### 📍 Highlight Timeline")
                st.caption("Detected highlight segments across the match duration")

                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use('Agg')

                fig, ax = plt.subplots(figsize=(14, 2.5))
                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#1a1a2e')

                for evt in selected_events:
                    start_min = evt['start'] / 2 / 60
                    end_min = evt['end'] / 2 / 60
                    conf = evt['confidence']
                    # Color intensity based on confidence
                    color_intensity = 0.4 + 0.6 * conf
                    ax.barh(0, end_min - start_min, left=start_min,
                            height=0.5, color=(0.39, 0.40, 0.95, color_intensity),
                            edgecolor='#8b8bff', linewidth=0.8)

                ax.set_xlim(0, video_duration_min)
                ax.set_yticks([])
                ax.set_xlabel('Match Time (minutes)', color='#a0a0cc', fontsize=10)
                ax.tick_params(colors='#a0a0cc')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_color('#2a2a4a')
                ax.grid(True, axis='x', alpha=0.15, color='#4a4a6a')

                st.pyplot(fig)
                plt.close(fig)

                # Detection details table (NO class labels, just timestamps and confidence)
                st.markdown("#### 📋 Detection Details")
                table_data = []
                for i, evt in enumerate(selected_events):
                    start_sec = evt['start'] / 2
                    end_sec = evt['end'] / 2
                    start_str = f"{int(start_sec // 60)}:{int(start_sec % 60):02d}"
                    end_str = f"{int(end_sec // 60)}:{int(end_sec % 60):02d}"
                    duration = end_sec - start_sec
                    table_data.append({
                        "Clip #": i + 1,
                        "Start Time": start_str,
                        "End Time": end_str,
                        "Duration": f"{duration:.0f}s",
                        "Confidence": f"{evt['confidence']:.1%}",
                    })

                st.dataframe(
                    table_data,
                    use_container_width=True,
                    hide_index=True,
                )

            else:
                st.error("❌ Failed to stitch video. Check that the video file matches the features.")
        else:
            progress_bar.progress(1.0, text="⚠️ No highlights detected at this threshold.")
            st.warning(
                f"No events exceeded the {threshold:.0%} confidence threshold. "
                "Try lowering the Detection Sensitivity slider in the sidebar."
            )
else:
    # Placeholder when files aren't uploaded yet
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; color: #6a6a9a;">
        <h2 style="color: #8b8bba; margin-bottom: 1rem;">👆 Upload a video to get started</h2>
        <p style="font-size: 1.1rem; max-width: 600px; margin: 0 auto; line-height: 1.8;">
            Upload your original <code>.mp4</code> video to extract features automatically.<br><br>
            <strong>Live Demo Mode:</strong> Upload a pre-extracted <code>.npy</code> file alongside your video to skip the heavy extraction and generate highlights instantly!
        </p>
    </div>
    """, unsafe_allow_html=True)
