# ⚡ Fast Mode: NPY Feature Extraction Guide

To avoid waiting for the heavy ResNet-152 model to process your video during a live demo or on a slow computer, you can extract the features ahead of time using Google Colab's free GPUs!

When you upload the `.npy` file alongside your `.mp4` video into the MatchCut AI app, it bypasses the heavy processing and generates highlights instantly (in about 5 seconds).

### How to Extract `.npy` Features using Google Colab

1. **Open Google Colab:** Go to [colab.research.google.com](https://colab.research.google.com/).
2. **Create a New Notebook.**
3. **Enable GPU:** Go to `Runtime` -> `Change runtime type` and select `T4 GPU`.
4. **Upload the Script:** Upload the `npy_extraction.py` script (found in this repository) to your Colab environment.
5. **Upload your Video:** Upload the `.mp4` video you want to process into the Colab environment (e.g., `match.mp4`).
6. **Run the Script:**
   Create a new cell in Colab and run the following command:
   ```bash
   !python npy_extraction.py --video match.mp4 --output match_features.npy
   ```
7. **Download the Features:** Once it finishes (it should only take a few seconds on a GPU), download the generated `match_features.npy` file.

### Using it in MatchCut AI
Now simply open the Streamlit app, upload your original `.mp4` video, and upload your new `match_features.npy` file into the "Pre-Extracted Features" box. Click Generate!
