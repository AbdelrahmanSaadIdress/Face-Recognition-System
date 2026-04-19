from src.config import DataConfig, PreprocessingConfig, load_config

if __name__ == "__main__":
    # Quick test to verify dataset loading and preprocessing.
    # Run this after making changes to lfw_dataset.py or preprocessing.py.

    import logging
    from pathlib import Path

    import pandas as pd
    import torch

    from src.data.lfw_dataset import LFWPairsDataset
    logging.basicConfig(level=logging.INFO)

    # Load configs
    config_path = Path("configs/base.yaml")
    config = load_config(config_path)
    data_cfg = config.data
    
    dataset = LFWPairsDataset(
        data_cfg=data_cfg,
        preproc_cfg=config.preprocessing,
        split="test",
    )

    # Test loading a few samples
    print((dataset._pairs.head()))  # Show first few pairs from CSV

    for i in range(5):
        img1, img2, label = dataset.__getitem__(i)
        print(f"Sample {i}: img1 shape={img1.shape}, img2 shape={img2.shape}, label={label}")
        import matplotlib.pyplot as plt
        # Visualise the first pair
        img1_np = img1.transpose(1, 2, 0)  # CxHxW → HxWxC
        img2_np = img2.transpose(1, 2, 0)
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(img1_np)
        axes[0].set_title("Image 1")
        axes[0].axis("off")
        axes[1].imshow(img2_np)
        axes[1].set_title("Image 2")
        axes[1].axis("off")
        plt.suptitle(f"Label: {'Same' if label == 1 else 'Different'}")
        plt.show()
        