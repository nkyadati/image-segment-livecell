import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import traceback

class LIVECellSegDataset(Dataset):
    """
    Dataset class for LIVECell semantic segmentation.
    Handles both Hugging Face image processors and raw tensor preparation.
    """

    def __init__(self, img_dir, mask_dir, processor=None, size=(512, 512)):
        """
        Args:
            img_dir (str): Path to input images.
            mask_dir (str): Path to ground truth masks.
            processor (transformers.ImageProcessor, optional): Hugging Face processor for SegFormer.
            size (tuple): Resize target (H, W).
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.size = size
        self.image_filenames = sorted([
            f for f in os.listdir(img_dir) if f.endswith(".tif")
        ])

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """
        Load image and mask for the given index.
        Returns a dictionary with pixel_values and labels.
        """
        try:
            image_filename = self.image_filenames[idx]
            mask_filename = image_filename.replace(".tif", "_mask.png")

            img_path = os.path.join(self.img_dir, image_filename)
            mask_path = os.path.join(self.mask_dir, mask_filename)

            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask, dtype=np.uint8)

            if self.processor:
                inputs = self.processor(images=image, return_tensors="pt", size=self.size)
                inputs["labels"] = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
                return {k: v.squeeze(0) for k, v in inputs.items()}
            else:
                image = image.resize(self.size)
                mask = Image.fromarray(mask).resize(self.size, Image.NEAREST)
                image = np.array(image).transpose(2, 0, 1) / 255.0
                mask = np.array(mask)
                return {
                    "pixel_values": torch.tensor(image, dtype=torch.float32),
                    "labels": torch.tensor(mask, dtype=torch.long),
                }

        except Exception as e:
            print(f"Failed to load/transform sample at index {idx}: {e}")
            traceback.print_exc()
            return {
                "pixel_values": torch.zeros((3, *self.size), dtype=torch.float32),
                "labels": torch.zeros(self.size, dtype=torch.long)
            }