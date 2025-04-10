from model import SegFormerModelWrapper
from config import Config
from PIL import Image
import torch
import numpy as np

class SegFormerInference:
    """
    Wrapper class for loading a trained SegFormer model and running inference on input images.
    """

    def __init__(self):
        """
        Initialize the inference class.

        Loads configuration, initializes the SegFormer model, and sets it to evaluation mode.
        """
        self.cfg = Config()
        self.model_wrapper = SegFormerModelWrapper(
            model_path=self.cfg.MODEL_SAVE_PATH,
            device=self.cfg.DEVICE,
            model_type=self.cfg.MODEL_TYPE
        )
        self.model_wrapper.model.eval()

    def predict(self, image: Image.Image) -> Image.Image:
        """
        Run inference on a single input image and return the predicted segmentation mask as a PIL Image.

        Args:
            image (Image.Image): Input RGB image.

        Returns:
            Image.Image: Predicted binary or multi-class segmentation mask as a PIL Image.
        """
        processor = self.model_wrapper.processor
        inputs = processor(images=image, return_tensors="pt", size=(512, 512)).to(self.cfg.DEVICE)

        with torch.no_grad():
            outputs = self.model_wrapper.model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

        # Convert predicted mask to PIL image
        mask_image = Image.fromarray((preds * 255).astype(np.uint8))
        return mask_image