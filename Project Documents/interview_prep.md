# Football Highlights Detection: Interview Prep Guide

This document provides a comprehensive overview of your project, breaking down the architecture, technical decisions, limitations, and a robust set of potential interview questions.

## 1. Project Overview
This project is an end-to-end deep learning pipeline that automatically generates highlight reels from full-length football match videos. It detects key events (Goals, Cards, and Substitutions) using models trained on the SoccerNet dataset and stitches the best clips into a compact video.

## 2. System Architecture (Two-Stage Pipeline)
The system uses a decoupled two-stage design because end-to-end training (raw pixels to event labels) requires immense GPU memory (40+ GB), which is not feasible on consumer hardware like Colab's free T4 GPU (15GB).

*   **Stage 1: Spatial Feature Extraction (ResNet-152 + PCA)**
    *   **What it does:** Processes raw video frames into mathematical representations.
    *   **Details:** Uses ResNet-152 to extract 2048-dimensional features, which are then reduced to 512 dimensions using PCA (Principal Component Analysis). This perfectly matches the feature extraction pipeline provided by the SoccerNet dataset, avoiding **Domain Shift** (which happened when you initially tried to use ResNet-18).
*   **Stage 2: Temporal Modeling (BiLSTM)**
    *   **What it does:** Understands sequences of frames to classify events over time.
    *   **Details:** Uses a Bidirectional LSTM (BiLSTM) with a sequence length of 30 frames (15 seconds at 2 FPS). The BiLSTM looks at both past and future frames to make a decision for the center frame (e.g., seeing player celebration *after* the goal helps confirm it). It outperforms the Transformer model on this dataset.
*   **Stage 3: Event Selection (0/1 Knapsack Algorithm)**
    *   **What it does:** Selects the best events to include in the final video.
    *   **Details:** Rather than just picking the top-N events (greedy approach), it uses the Dynamic Programming 0/1 Knapsack algorithm to maximize the total "value" (confidence of detection) while staying under the "capacity" limit (a 10-minute highlight budget).
*   **Stage 4: Output (MoviePy Stitcher)**
    *   **What it does:** Cuts and concatenates video clips (H.264 codec, AAC audio) into the final highlight reel.

## 3. Training & Optimization Details
*   **Dataset:** SoccerNet v2 Action Spotting benchmark. You trained on a 60-match subset (~120 hours) because of Colab's 112GB disk limit (the full dataset is 500 matches / ~1000 hours).
*   **Classes:** 4 classes (Goal, Cards, Substitution, Background). You merged the original 17 classes into 4 to prevent severe overfitting on the sparse dataset.
*   **Class Imbalance:** 99% of frames are "Background." You solved this using **inverse frequency class weighting** in `CrossEntropyLoss` (Background weight ~0.26, Goal weight ~12.5), penalizing event misclassifications heavily.
*   **Length Regularization:** Added a penalty term to ensure the model targets predicting events roughly 15% of the time, preventing it from over-predicting events just to avoid missing them.
*   **Optimizer & Gradient Clipping:** Used Adam optimizer (stable for severe class imbalance) and `clip_grad_norm_` set to 5.0 to prevent LSTM exploding gradients.
*   **Learning Rate Scheduler:** `ReduceLROnPlateau` drops the LR when validation mAP plateaus for 5 epochs.

## 4. Results
*   **BiLSTM Model:** Achieved **96.3% Accuracy** and **35.1% mAP (Mean Average Precision)**.
*   **Why is Accuracy so high but mAP lower?** Predicting "Background" all the time gives 95% accuracy. mAP measures ranking quality for the actual events (ignoring Background). 35.1% mAP is excellent given you only used 12% of the training data (SOTA is ~70% with 100% data and massive compute).
*   **Transformer Performance:** Underperformed (8.9% mAP) because Transformers need massive datasets to learn attention patterns and do better on very long sequences, whereas the BiLSTM has a built-in inductive bias for short sequential data (30 frames).

## 5. Potential Interview Questions

### Architecture & Design
1.  **Why a two-stage pipeline instead of end-to-end?**
    *   *Ans:* End-to-end requires 40+ GB GPU RAM. Decoupling spatial extraction and temporal modeling makes it feasible on a free 15GB Colab GPU.
2.  **Why ResNet-152 + PCA?**
    *   *Ans:* It matches the official SoccerNet feature pipeline (2048-dim reduced to 512). Initially, we used ResNet-18, which created a domain shift (mismatched feature spaces). Upgrading to ResNet-152 fixed this.
3.  **Why BiLSTM over a standard LSTM?**
    *   *Ans:* Standard LSTMs only use past context. BiLSTMs use past and future context. Seeing a celebration 3 seconds *after* an event helps confirm it was a goal.
4.  **Why did the Transformer underperform?**
    *   *Ans:* 1) Data scale (Transformers need way more data than 60 matches). 2) Sequence length (Transformers shine on 1000+ frames, LSTMs are fine for 30). 3) Positional encoding overhead acts as noise with limited data.
5.  **What is the Knapsack algorithm used for?**
    *   *Ans:* Maximizing total highlight value (detection confidence) under a strict time limit (budget). It's mathematically optimal compared to greedy selection.

### Training & Optimization
6.  **How did you handle class imbalance (99% Background)?**
    *   *Ans:* Inverse frequency class weighting in CrossEntropyLoss, meaning misclassifying a goal penalizes the model 50x more than misclassifying background.
7.  **What is length regularization?**
    *   *Ans:* A penalty targeting a ~15% event rate, preventing the model from predicting events everywhere just to be safe.
8.  **Why use Gradient Clipping?**
    *   *Ans:* LSTMs are prone to exploding gradients (multiplicative accumulation through time). Clipping to an L2 norm of 5.0 prevents NaNs and training failure.

### Evaluation & Results
9.  **What does 35.1% mAP mean?**
    *   *Ans:* It's the Mean Average Precision. It means our model correctly ranks about 1 in 3 event frames above background frames. We achieved 50% of SOTA performance using only 12% of the data.
10. **Why is test accuracy 96% but mAP only 35%?**
    *   *Ans:* The dataset is 95% background. Accuracy is skewed by the background class, while mAP strictly measures how well we rank rare events.

### Domain Knowledge & Future Work
11. **Explain the Domain Shift problem you encountered.**
    *   *Ans:* We trained on ResNet-152 features from SoccerNet but initially used ResNet-18 for inference on custom videos. The feature distributions were fundamentally incompatible. We fixed it by using ResNet-152 for inference too.
12. **What would you do differently with unlimited compute?**
    *   *Ans:* Train on all 500 matches, add an audio stream (crowd noise correlates with goals), and implement a learned domain adaptation layer.
13. **How does this differ from commercial systems?**
    *   *Ans:* Systems like WSC Sports use 10,000+ matches, multi-modal inputs (audio/video/text overlays), and humans-in-the-loop. Our system proves the same core algorithmic pipeline works, just constrained by data and compute.
