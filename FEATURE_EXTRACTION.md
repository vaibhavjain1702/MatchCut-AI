# ⚡ Feature Extraction Guide

To avoid waiting for the ResNet-152 model to process your video locally during a live demo, you can extract the `.npy` features ahead of time using Google Colab.

### How to Extract Features

1. Open `Highlight_generation_FINAL_colab.ipynb` in Google Colab.
2. Enable GPU: Go to `Runtime` -> `Change runtime type` and select `T4 GPU`.
3. Run the setup cells to load the models.
4. Upload your `.mp4` video to Colab.
5. Run the feature extraction cell. The Colab notebook will generate a `.npy` file containing the 512-dim PCA features.
6. Download the `.npy` file to your local machine.

### Fast Mode in MatchCut AI
Upload your `.mp4` video and the downloaded `.npy` file into the Streamlit app. The app will bypass the heavy local processing and generate highlights instantly!
