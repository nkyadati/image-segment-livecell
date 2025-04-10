import os
import torch
import torch.nn.functional as F
import numpy as np
import traceback
from sklearn.metrics import jaccard_score
from tqdm import tqdm 


class SegFormerTrainer:
    """
    Trainer class to handle training and validation for SegFormer on multi-class segmentation.
    """

    def __init__(self, model, optimizer, scheduler, config, logger):
        """
        Initialize the trainer.

        Args:
            model: SegFormer model object.
            optimizer (torch.optim.Optimizer): Optimizer instance.
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
            config (Config): Configuration object.
            logger: Logger instance.
        """
        self.model = model.get_model()
        self.wrapper = model
        self.processor = model.processor
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.logger = logger
        self.device = config.DEVICE
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def compute_per_class_iou(self, preds, labels, num_classes):
        """
        Compute IoU for each class and return dictionary of IoUs.

        Args:
            preds (np.ndarray): Predicted masks [B, H, W]
            labels (np.ndarray): Ground truth masks [B, H, W]
            num_classes (int): Total number of classes

        Returns:
            Tuple[dict, float]: (Per-class IoUs, mean IoU over present classes)
        """
        ious = {}
        valid_ious = []

        for cls in range(num_classes):
            pred_cls = (preds == cls).astype(np.uint8)
            label_cls = (labels == cls).astype(np.uint8)

            if label_cls.sum() == 0:
                ious[cls] = None  # Class not present in GT
                continue

            iou = jaccard_score(label_cls.flatten(), pred_cls.flatten(), average='binary')
            ious[cls] = iou
            valid_ious.append(iou)

        mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
        return ious, mean_iou

    def validate(self, dataloader):
        """
        Validate the model and return mean IoU and loss.
        Also logs per-class IoU scores.

        Args:
            dataloader (DataLoader): Validation or test dataloader.

        Returns:
            Tuple[float, float]: Mean IoU, Mean loss
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        try:
            with torch.no_grad():
                for batch in dataloader:
                    pixel_values = batch["pixel_values"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = self.model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss
                    total_loss += loss.item()

                    preds = torch.argmax(outputs.logits, dim=1)

                    if preds.shape[-2:] != labels.shape[-2:]:
                        labels = F.interpolate(
                            labels.unsqueeze(1).float(),
                            size=preds.shape[-2:],
                            mode="nearest"
                        ).squeeze(1).long()

                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())

            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            # Compute per-class IoUs
            ious = {}
            valid_ious = []
            for cls in range(self.config.NUM_CLASSES):
                pred_cls = (all_preds == cls).astype(np.uint8)
                label_cls = (all_labels == cls).astype(np.uint8)

                if label_cls.sum() == 0:
                    ious[cls] = None
                    self.logger.info(f"Class {cls}: IoU = N/A (not present)")
                    continue

                iou = jaccard_score(label_cls.flatten(), pred_cls.flatten(), average='binary')
                ious[cls] = iou
                valid_ious.append(iou)
                self.logger.info(f"Class {cls}: IoU = {iou:.4f}")

            mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
            mean_loss = total_loss / len(dataloader)

            self.logger.info(f"Validation - mIoU: {mean_iou:.4f}, Loss: {mean_loss:.4f}")
            return mean_iou, mean_loss

        except Exception as e:
            self.logger.error("Error during validation.")
            self.logger.error(traceback.format_exc())
            return 0.0, float("inf")

    def train(self, train_loader, val_loader, start_epoch=0):
        """
        Run the training loop with early stopping and logging.
        
        Args:
            train_loader (DataLoader): Training set loader.
            val_loader (DataLoader): Validation set loader.
            start_epoch (int): Epoch to resume training from.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(start_epoch, self.config.NUM_EPOCHS):
            self.model.train()
            total_train_loss = 0

            self.logger.info(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")

            try:
                progress_bar = tqdm(train_loader, desc="Training", leave=False)

                for batch in progress_bar:
                    pixel_values = batch["pixel_values"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = self.model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    total_train_loss += loss.item()
                    progress_bar.set_postfix({"Loss": loss.item()})

                avg_train_loss = total_train_loss / len(train_loader)
                val_miou, val_loss = self.validate(val_loader)
                self.scheduler.step(val_loss)

                self.logger.info(
                    f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}"
                )

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.wrapper.save_pretrained(self.config.MODEL_SAVE_PATH)
                    self.logger.info("New best model saved.")
                else:
                    patience_counter += 1
                    self.logger.info(f"No improvement. Patience {patience_counter}/{self.config.PATIENCE}")

                # Save checkpoint
                torch.save({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "patience_counter": patience_counter,
                }, self.config.CHECKPOINT_PATH)

                if patience_counter >= self.config.PATIENCE:
                    self.logger.info("Early stopping triggered.")
                    break

            except Exception as e:
                self.logger.error("Error during training loop.")
                self.logger.error(traceback.format_exc())
                break