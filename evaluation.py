import os
import torch
import numpy as np
import random
import time
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from evaluation_metrics import (
    compute_iou,
    compute_dice,
    compute_pixel_accuracy,
    compute_mean_pixel_accuracy,
    compute_boundary_dice,
)

from utils import visualize_triplet_images


def load_images(image_dir, mask_dir):
    """
    Load image and mask file pairs from given directories.

    Args:
        image_dir (str): Path to the directory containing input images.
        mask_dir (str): Path to the directory containing ground truth masks.

    Returns:
        List[Tuple[str, str]]: List of tuples with image and corresponding mask paths.
    """
    files = sorted([f for f in os.listdir(image_dir) if f.endswith(".tif")])
    samples = []
    for f in files:
        image_path = os.path.join(image_dir, f)
        mask_path = os.path.join(mask_dir, f.replace(".tif", "_mask.png"))
        if os.path.exists(image_path) and os.path.exists(mask_path):
            samples.append((image_path, mask_path))
    return samples


def evaluate_on_test_set(model_path, test_img_dir, test_mask_dir, device="cuda"):
    """
    Evaluate a trained SegFormer model on the LIVECell test set using various metrics.

    Loads images and masks, runs inference, computes metrics (IoU, Dice, Accuracy),
    and optionally saves triplet visualizations (input, ground truth, prediction).

    Args:
        model_path (str): Path to the trained SegFormer model directory.
        test_img_dir (str): Directory path containing test input images.
        test_mask_dir (str): Directory path containing test ground truth masks.
        device (str): Device to run inference on ("cuda" or "cpu").

    Returns:
        None
    """
    processor = SegformerImageProcessor.from_pretrained(model_path)
    model = SegformerForSemanticSegmentation.from_pretrained(model_path).to(device)
    model.eval()

    samples = load_images(test_img_dir, test_mask_dir)

    all_preds = []
    all_gts = []

    # Randomly select indices for visualization
    random.seed(42)
    selected_indices = set(random.sample(range(len(samples)), min(10, len(samples))))
    total_time = 0.0

    for idx, (img_path, mask_path) in enumerate(tqdm(samples, desc="Evaluating")):
        image = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path).convert("L"))

        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze(0).cpu()

        h, w = mask.shape
        logits_resized = torch.nn.functional.interpolate(
            logits.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(0)

        pred = torch.argmax(logits_resized, dim=0).numpy()

        all_preds.append(pred)
        all_gts.append(mask)

        # Save triplet visualizations only for selected random indices
        if idx in selected_indices:
            os.makedirs("triplets", exist_ok=True)
            save_path = os.path.join("triplets", f"triplet_{idx + 1}.png")
            visualize_triplet_images(image, mask, pred, save_path=save_path, show=False)

    y_pred = np.concatenate([p.flatten() for p in all_preds])
    y_true = np.concatenate([g.flatten() for g in all_gts])

    print("Evaluation Metrics:")
    miou, ious = compute_iou(y_true, y_pred)
    print(f"Mean IoU: {miou:.4f}")
    print(f"Per-class IoU: {np.round(ious, 4)}")

    dice_mean, dice_scores = compute_dice(y_true, y_pred)
    print(f"Mean Dice: {dice_mean:.4f}")
    print(f"Per-class Dice: {np.round(dice_scores, 4)}")

    acc = compute_pixel_accuracy(y_true, y_pred)
    print(f"Pixel Accuracy: {acc:.4f}")

    mpa = compute_mean_pixel_accuracy(y_true, y_pred)
    print(f"Mean Pixel Accuracy: {mpa:.4f}")

    b_dice, b_scores = compute_boundary_dice(
        np.concatenate(all_gts), np.concatenate(all_preds), radius=2
    )
    print(f"Boundary Dice: {b_dice:.4f}")
    print(f"Per-class Boundary Dice: {np.round(b_scores, 4)}")