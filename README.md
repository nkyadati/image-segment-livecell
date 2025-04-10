# SegFormer-Based Semantic Segmentation on LIVECell

This project provides an end-to-end pipeline for **multi-class semantic segmentation** of the LIVECell dataset using **SegFormer** from HuggingFace Transformers.  
Built with full support for training, evaluation, visualization, and API deployment.

---

## What is LIVECell?

[LIVECell](https://github.com/sartorius-research/LIVECell) is a large-scale dataset for label-free segmentation of cells in microscopy images.

Key properties:

- 8 cell types + background (9 classes)
- Brightfield microscopy images
- COCO-style annotations (used to generate masks)
- 520x704 resolution
- Over 1.6 million annotated cells

---

## Features

- ✅ SegFormer-based architecture (`nvidia/segformer-b0-finetuned-ade-512-512`)
- ✅ Multi-class segmentation (9 LIVECell cell types)
- ✅ Full training loop with early stopping, checkpointing, and LR scheduler
- ✅ Automatic test set evaluation with:
  - 📊 Mean IoU + per-class IoU
  - 📊 Dice Score + per-class Dice
  - 📊 Pixel Accuracy & Mean Pixel Accuracy
  - 📊 Boundary-aware Dice (boundary sensitivity)
- ✅ Visualizations for 10 random test images
- ✅ Supports both CLI and headless environments (e.g., AWS EC2, Colab)
- ✅ FastAPI-based inference endpoint
- ✅ Dockerized for reproducibility

---

## Project Structure

```
image-segment-livecell/
├── main.py                   # Entry point with training/evaluation CLI
├── config.py                 # All paths and hyperparameters
├── model.py                  # SegFormer model wrapper
├── trainer.py                # Training pipeline
├── dataset.py                # LIVECell dataset handler
├── evaluation.py             # Evaluation on test set + plots
├── evaluation_metrics.py     # 5 semantic segmentation metrics
├── visualize.py              # Triplet image visualizer
└── triplets/                 # Saved prediction triplets
```

```
app/
├── main.py                   # FastAPI app for prediction
├── inference.py              # Inference wrapper (PyTorch)
├── inference_onnx.py         # Optional ONNX inference wrapper
├── utils.py                  # Base64 + image conversion helpers
```

---
## Clone the repository

```bash
git clone https://github.com/nkyadati/image-segment-livecell.git
cd image-segment-livecell
```

## Create Conda Environment

We recommend using a Conda environment to manage dependencies.

### Steps:

```bash
# 1. Create environment
conda create -n segformer-env python=3.9 -y

# 2. Activate environment
conda activate segformer-env

# 3. Install requirements
pip install -r requirements.txt
```

If you're using a CPU-only setup, replace the PyTorch line with:

```bash
pip install torch torchvision torchaudio
```
---

## Download the dataset ad the trained model

You can download the dataset from [Google Drive](https://drive.google.com/drive/folders/1U54JlVyxhflFo0fUOr7Zlo0uh7aHg_kN?usp=drive_link) and place it in the folder: `image-segment-livecell`. You should see a folder structure like below:

```
image-segment-livecell/
├── dataset
   ├── images
      ├── train
      ├── val
      ├── test
   ├── annotations
      ├── train
      ├── val
      ├── test
```

You can download the trained model from [Google Drive](https://drive.google.com/drive/folders/1ocRpsT0ndKx3kUKq3_2Ty7rXpWFyvh4M?usp=drive_link) and place it in the folder: `image-segment-livecell`. You should see a folder structure like below:

```
image-segment-livecell/
├── trained_model
   ├── checkpoint.pth
   ├── config.json
   ├── model.safetensors
   ├── preprocessor_config.json
```

You can update the paths in `image-segment-livecell/config.py` 

---

## Run Evaluation

```bash
python main.py --eval_only
```

Train + Evaluate:

```bash
python main.py --eval
```

Only train:

```bash
python main.py
```

Python Notebook for training and evaluation: SegFormer_Training_LIVECell.ipynb

---
## Evaluation Metrics

Metrics evaluated on the test set:

- **Mean IoU** + Per-Class IoU  
- **Dice Score** + Per-Class Dice  
- **Pixel Accuracy**  
- **Mean Pixel Accuracy**  
- **Boundary Dice** + Per-Class Boundary Dice  

Plots and logs are saved automatically.

---

## Performance Metrics on LIVECell Test Set


| **Metric**              | **Score**   |
|-------------------------|-------------|
| **Mean IoU**          | `0.85`     |
| **Dice Score**        | `0.92`     |
| **Pixel Accuracy**    | `0.95`     |
| **Mean Pixel Accuracy** | `0.95`  |
| **Boundary Dice Score** | `0.57` |

---

### Per-Class IoU Scores

| Class ID | Cell Type (optional) | IoU Score |
|----------|----------------------|-----------|
| 0        | Background           | `0.91`    |
| 1        | SHSY5Y               | `0.93`    |
| 2        | SkBr3                | `0.80`    |
| 3        | A172                 | `0.76`    |
| 4        | BV2                  | `0.85`    |
| 5        | BT474                | `0.84`    |
| 6        | Huh7                 | `0.76`    |
| 7        | LNCaP                | `0.93`    |
| 8        | RAW264.7             | `0.89`    |

---

### Inference speed

The average inference speed on the test set of ~1500 images : 55 ms

## Visualization

- 10 randomly chosen test images are visualized with:
  - Original image
  - Ground-truth mask
  - Predicted mask

Saved to: `image-segment-livecell/triplets/`

---

## Web App Deployment

### Build the Docker image (Run these command in the root directory):

```bash
docker build -t segformer-livecell .
```

### Run the container:

```bash
docker run -p 8000:8000 segformer-livecell
```

### cURL Prediction Request:

```bash
curl -X POST "http://localhost:8000/predict/" \
     -F "file=@/path/to/image.tif" --output output_mask.png
```

---
## Potential improvements

1. Boundary DICE score is low for the trained model, it would worth to try a boundary aware loss function while training and a bigger model for the encoder, like nvidia/mit-b5

2. Current model only tackles semantic segmentation, newer algorithms provide a unified framework for semantic, instance, and panoptic segmentation (e.g., Mask2Former)

3. Converting the PyTorch model to ONNX could aid in better inference speeds.

4. Investigate post-processing methods for the model outputs: CRF, Guilded Filtering etc. 

5. Compare the performance of transformer based methods to CNN based methods (e.g., UNet)

.
