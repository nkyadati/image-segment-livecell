import os
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

class SegFormerModelWrapper:
    """
    A wrapper class for loading, saving, and using the SegFormer model
    for multi-class semantic segmentation (LIVECell: 9 classes).
    """

    def __init__(self, model_path: str, device: torch.device, model_type: str, num_classes: int = 9):
        """
        Initialize the SegFormer model and processor.

        Args:
            model_path (str): Directory to load/save the model.
            device (torch.device): Torch device (CPU or CUDA).
            num_classes (int): Number of segmentation classes.
        """
        self.device = device
        self.model_path = model_path
        self.num_classes = num_classes
        self.model_type = model_type

        if os.path.exists(os.path.join(model_path, "config.json")):
            print("Loading model from checkpoint...")
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_path).to(device)
            self.processor = SegformerImageProcessor.from_pretrained(model_path)
        else:
            print("Initializing new SegFormer model...")
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                self.model_type,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            ).to(device)
            self.processor = SegformerImageProcessor.from_pretrained(self.model_type)

        # Optionally compile for speed (PyTorch 2.0+)
        try:
            self.model = torch.compile(self.model)
        except Exception:
            pass

    def save_pretrained(self, path):
        """
        Save the model and processor to the given path in HuggingFace format.
        """
        try:
            self.model.save_pretrained(path)
            self.processor.save_pretrained(path)
            print(f"Model and processor saved to: {path}")
        except Exception as e:
            print(f"Failed to save model: {e}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model weights from a checkpoint file.

        Args:
            checkpoint_path (str): Path to .pth checkpoint file.
        """
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict["model_state"])

    def predict(self, image):
        """
        Run inference on a single image.

        Args:
            image (PIL.Image): Input RGB image.

        Returns:
            np.ndarray: Predicted mask (H, W) with class labels.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.squeeze(0).cpu()
            logits_resized = torch.nn.functional.interpolate(
                logits.unsqueeze(0), size=image.size[::-1], mode="bilinear", align_corners=False
            ).squeeze(0)
            pred_mask = torch.argmax(logits_resized, dim=0).numpy()
        return pred_mask

    def get_model(self):
        """
        Return the underlying HuggingFace model object (e.g., for training).

        Returns:
            torch.nn.Module: SegFormer model instance.
        """
        return self.model