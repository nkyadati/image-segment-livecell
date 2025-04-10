import numpy as np
from sklearn.metrics import jaccard_score, f1_score, accuracy_score
from scipy.ndimage import binary_dilation

def compute_iou(gt, pred, num_classes=9):
    """
    Compute mean Intersection over Union (mIoU) and per-class IoU.

    Args:
        gt (np.ndarray): Ground truth segmentation mask.
        pred (np.ndarray): Predicted segmentation mask.
        num_classes (int): Number of classes.

    Returns:
        Tuple[float, np.ndarray]: Mean IoU and array of per-class IoUs.
    """
    gt = gt.flatten()
    pred = pred.flatten()
    ious = jaccard_score(gt, pred, average=None, labels=list(range(num_classes)))
    miou = np.mean(ious)
    return miou, ious

def compute_dice(gt, pred, num_classes=9):
    """
    Compute mean Dice score and per-class Dice scores.

    Args:
        gt (np.ndarray): Ground truth segmentation mask.
        pred (np.ndarray): Predicted segmentation mask.
        num_classes (int): Number of classes.

    Returns:
        Tuple[float, np.ndarray]: Mean Dice score and array of per-class Dice scores.
    """
    gt = gt.flatten()
    pred = pred.flatten()
    dice_scores = f1_score(gt, pred, average=None, labels=list(range(num_classes)))
    mean_dice = np.mean(dice_scores)
    return mean_dice, dice_scores

def compute_pixel_accuracy(gt, pred):
    """
    Compute overall pixel-wise accuracy.

    Args:
        gt (np.ndarray): Ground truth segmentation mask.
        pred (np.ndarray): Predicted segmentation mask.

    Returns:
        float: Pixel accuracy score.
    """
    return accuracy_score(gt.flatten(), pred.flatten())

def compute_mean_pixel_accuracy(gt, pred, num_classes=9):
    """
    Compute mean pixel accuracy over all present classes.

    Args:
        gt (np.ndarray): Ground truth segmentation mask.
        pred (np.ndarray): Predicted segmentation mask.
        num_classes (int): Number of classes.

    Returns:
        float: Mean pixel accuracy across classes.
    """
    accs = []
    for c in range(num_classes):
        gt_c = (gt == c).astype(int)
        pred_c = (pred == c).astype(int)
        if np.sum(gt_c) > 0:
            accs.append(np.sum((gt_c == 1) & (pred_c == 1)) / np.sum(gt_c))
    return np.mean(accs) if accs else 0.0

def compute_boundary_dice(gt, pred, radius=2, num_classes=9):
    """
    Compute mean and per-class boundary Dice scores.

    Args:
        gt (np.ndarray): Ground truth segmentation mask.
        pred (np.ndarray): Predicted segmentation mask.
        radius (int): Radius used for boundary dilation.
        num_classes (int): Number of classes.

    Returns:
        Tuple[float, List[float]]: Mean boundary Dice score and per-class boundary Dice scores.
    """
    def boundary_map(mask):
        dilated = binary_dilation(mask, iterations=radius)
        eroded = binary_dilation(mask == 0, iterations=radius)
        return dilated & eroded

    dice_scores = []
    for cls in range(num_classes):
        gt_mask = (gt == cls)
        pred_mask = (pred == cls)

        gt_boundary = boundary_map(gt_mask)
        pred_boundary = boundary_map(pred_mask)

        intersection = np.logical_and(gt_boundary, pred_boundary).sum()
        total = gt_boundary.sum() + pred_boundary.sum()

        if total == 0:
            dice_scores.append(1.0)  # Perfect match if both empty
        else:
            dice_scores.append(2.0 * intersection / total)

    return np.mean(dice_scores), dice_scores