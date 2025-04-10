import argparse
import torch
from trainer import SegFormerTrainer
from model import SegFormerModelWrapper
from dataset import LIVECellSegDataset
from torch.utils.data import DataLoader
from config import cfg, logger
from evaluation import evaluate_on_test_set

def main():
    parser = argparse.ArgumentParser(description="SegFormer LIVECell Pipeline")
    parser.add_argument('--eval', action='store_true', help='Run evaluation on test set after training')
    parser.add_argument('--eval_only', action='store_true', help='Run evaluation only without training')
    args = parser.parse_args()

    logger.info("Starting SegFormer training pipeline")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_wrapper = SegFormerModelWrapper(cfg.MODEL_SAVE_PATH, device, cfg.MODEL_TYPE)

    if args.eval_only:
        logger.info("Running evaluation only mode...")
        evaluate_on_test_set(
            model_path=cfg.MODEL_SAVE_PATH,
            test_img_dir=cfg.TEST_IMG_DIR,
            test_mask_dir=cfg.TEST_MASK_DIR,
            device=device
        )
        return

    # Load Datasets
    train_dataset = LIVECellSegDataset(cfg.TRAIN_IMG_DIR, cfg.TRAIN_MASK_DIR, model_wrapper.processor)
    val_dataset = LIVECellSegDataset(cfg.VAL_IMG_DIR, cfg.VAL_MASK_DIR, model_wrapper.processor)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE)

    optimizer = torch.optim.AdamW(model_wrapper.model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    trainer = SegFormerTrainer(model_wrapper, optimizer, scheduler, cfg, logger)
    trainer.train(train_loader, val_loader)

    if args.eval:
        logger.info("Running evaluation after training")
        evaluate_on_test_set(
            model_path=cfg.MODEL_SAVE_PATH,
            test_img_dir=cfg.TEST_IMG_DIR,
            test_mask_dir=cfg.TEST_MASK_DIR,
            device=device
        )

if __name__ == "__main__":
    main()