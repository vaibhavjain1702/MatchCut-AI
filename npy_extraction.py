import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms
from sklearn.decomposition import PCA
import argparse
import os
import time

def extract_features(video_path, output_path, target_fps=2, batch_size=32):
    """
    Extracts ResNet-152 features from an mp4 video and saves them to a .npy file.
    Designed to be run on Google Colab (CUDA) for maximum speed.
    """
    print(f"🎬 Processing video: {video_path}")
    if not os.path.exists(video_path):
        print("❌ Video file not found.")
        return

    # Automatically use CUDA (Colab GPU) if available, else fallback
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("🚀 Using NVIDIA GPU (CUDA) - This will be extremely fast!")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("⚡ Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("🐌 Using CPU - This will be slow")

    print("📦 Loading ResNet-152...")
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
        return

    frame_interval = max(1, int(round(fps / target_fps)))
    raw_features = []
    batch_frames = []
    count = 0
    saved_count = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                batch_frames.append(preprocess(img))
                
                if len(batch_frames) == batch_size:
                    batch_tensor = torch.stack(batch_frames).to(device)
                    feat = feature_extractor(batch_tensor)
                    feat = feat.view(batch_size, -1).cpu().numpy()
                    raw_features.extend(feat)
                    saved_count += batch_size
                    batch_frames = []
                    print(f"   Extracted {saved_count} frames...")
                    
            count += 1
            
        # Process remainder
        if len(batch_frames) > 0:
            batch_tensor = torch.stack(batch_frames).to(device)
            feat = feature_extractor(batch_tensor)
            feat = feat.view(len(batch_frames), -1).cpu().numpy()
            raw_features.extend(feat)
            saved_count += len(batch_frames)
            print(f"   Extracted {saved_count} frames...")

    cap.release()
    extract_time = time.time() - start_time
    print(f"✅ Extraction complete in {extract_time:.1f}s")

    raw_features = np.array(raw_features)
    
    print("🔄 Applying PCA (2048 -> 512)...")
    if raw_features.shape[0] < 512:
        pca = PCA(n_components=raw_features.shape[0])
        features_reduced = pca.fit_transform(raw_features)
        features_512 = np.pad(features_reduced, ((0,0), (0, 512 - raw_features.shape[0])))
    else:
        pca = PCA(n_components=512)
        features_512 = pca.fit_transform(raw_features)
        
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    np.save(output_path, features_512.astype(np.float32))
    print(f"💾 Saved 512-dim features to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ResNet-152 features from video.")
    parser.add_argument("--video", required=True, help="Path to input .mp4 video")
    parser.add_argument("--output", required=True, help="Path to save output .npy file")
    parser.add_argument("--fps", type=int, default=2, help="Target FPS for extraction")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for GPU")
    
    args = parser.parse_args()
    extract_features(args.video, args.output, args.fps, args.batch)
