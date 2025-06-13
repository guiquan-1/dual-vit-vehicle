# dual-vit-vehicle

# Dual-Stream Vision Transformer for AAU RainSnow Dataset

This project implements a dual-stream Vision Transformer (ViT) architecture for segmenting challenging weather scenes using the AAU RainSnow dataset. It combines RGB and thermal streams to improve segmentation robustness under adverse weather.

This README also serves as an explanation for our framework 

---

## Project Structure

| File | Description |
|------|-------------|
| `aauRainSnow-rgb.json` | Annotation file for RGB images (COCO format). |
| `aauRainSnow-thermal.json` | Annotation file for thermal images. |
| `aauRainSnowUtility.py` | Utility functions for loading, preprocessing, and working with the AAU dataset. |
| `Dual Stream VIT (Main).ipynb` | Main training pipeline for dual-stream ViT using RGB and thermal data. |
| `dualstream_VIT_final.pth` | Final model checkpoint used for testing or visualization. |
| `README.md`
| `SegFormer Baseline.ipynb` | Baseline implementation using SegFormer (for comparison). |
| `splitVideoToFrames.bat` | Utility script to split videos into frames for dataset creation. |
| `Testing From Scratch.ipynb` | Training and evaluating from a fresh model initialization. |
| `Visualize AAU RainSnow annotations.ipynb` | Visualizes segmentation masks and annotations on RGB and thermal images. |
| `VIT Baseline + Comparison.ipynb` | Baseline single-stream ViT with Dual Stream ViT comparison and evaluation. |
| `VIT_Baseline.pth` | Trained checkpoint of baseline single-stream ViT model. |

---

## Key Code Explanations

### 🔸 `Dual Stream VIT (Main).ipynb`
- Implements two ViT encoders: one for RGB and one for thermal.
- Combines the two feature embeddings and passes them through a simple MLP decoder.
- Uses pretrained ViT weights for RGB, and partially loads them for the thermal stream (omitting input projection weights due to channel mismatch).
- **Note**: Look for the line that filters the projection layer:
  ```python
  if not k.startswith("embeddings.patch_embeddings.projection.")
- **Note**: Despite the segmentation class having 7 unique classes, the dataset actually has 9 non-unique segmentation classified as 'unknown':
 ''' Thus during model instantiation, we set the num_classes to 9, however during evaluation we only take the 7 known classes with proper label. Label 0 can be seen as background pixels

### 🔸 `VIT Baseline + Comparison.ipynb`
- Implements a simple baseline using the ViT-base Patch16-224.
- The baseline only processes one feature (RGB).
- Fully using ViT pretrained weights for the architecture.
- Pixel Accuracy, Mean Intersection over Union, and Per class Accuracy is done here.
- **Note**: Despite the segmentation class having 7 unique classes, the dataset actually has 9 non-unique segmentation classified as 'unknown':
 ''' Thus during model instantiation, we set the num_classes to 9, however during evaluation we only take the 7 known classes with proper label. Label 0 can be seen as background pixels

### 🔸 `Testing From Scratch.ipynb`
- Attempted to build a standalone ViT with only partially using the pretrained weights of a well-trained ViT model.
- Attempts at effectively distinguishing the segmentation objects on the images prove to be difficult as the dataset is not beginner friendly for any newly trained Trasnformer models.
- Partially used ViT weights are used for the RGB encoder part while the Thermal encoder is designed from scratch.
- **Note**: Despite the segmentation class having 7 unique classes, the dataset actually has 9 non-unique segmentation classified as 'unknown':
 ''' Thus during model instantiation, we set the num_classes to 9, however during evaluation we only take the 7 known classes with proper label. Label 0 can be seen as background pixels