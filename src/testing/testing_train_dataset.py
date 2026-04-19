from src.data import TrainFaceDataset
from src.config import DataConfig, PreprocessingConfig, load_config
from pathlib import Path

def test_train_dataset():
    config_path = Path("configs/base.yaml")
    config = load_config(config_path)
    data_cfg = config.data
    preproc_cfg = config.preprocessing

    dataset = TrainFaceDataset(data_cfg, preproc_cfg, split="train")
    print(f"Dataset length: {len(dataset)}")
    for i in range(360,380):
        print(dataset._samples[i])  # Accessing the first few samples to check for errors

    print(dataset._label_map)  # Accessing the first few samples to check for errors
    
    print("===="*20)



if __name__ == "__main__":
    test_train_dataset()    