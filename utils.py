import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
import numpy as np
from sklearn.metrics import jaccard_score

# 9-Class color palette for LIVECell
COLOR_LIST = [
    (230/255, 25/255, 75/255),    # Class 0 - Red
    (60/255, 180/255, 75/255),    # Class 1 - Green
    (255/255, 225/255, 25/255),   # Class 2 - Yellow
    (0/255, 130/255, 200/255),    # Class 3 - Blue
    (245/255, 130/255, 48/255),   # Class 4 - Orange
    (145/255, 30/255, 180/255),   # Class 5 - Purple
    (70/255, 240/255, 240/255),   # Class 6 - Cyan
    (240/255, 50/255, 230/255),   # Class 7 - Magenta
    (210/255, 245/255, 60/255)    # Class 8 - Lime
]
CUSTOM_CMAP = ListedColormap(COLOR_LIST)

def visualize_triplet_images(image, gt_mask, pred_mask, save_path=None, show=False):
    """
    Visualize original image, ground-truth mask, and predicted mask side by side.

    Args:
        image (PIL.Image or np.ndarray): Original RGB image
        gt_mask (np.ndarray): Ground-truth mask (H, W)
        pred_mask (np.ndarray): Predicted mask (H, W)
        save_path (str, optional): If provided, saves the figure to this path
        show (bool, optional): If True, displays the plot
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(image)
    axs[0].set_title("Input Image")
    axs[0].axis("off")

    axs[1].imshow(gt_mask, cmap=CUSTOM_CMAP, vmin=0, vmax=8)
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")

    axs[2].imshow(pred_mask, cmap=CUSTOM_CMAP, vmin=0, vmax=8)
    axs[2].set_title("Predicted Mask")
    axs[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()
    
    plt.close(fig)