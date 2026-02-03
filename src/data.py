import pandas as pd
from datasets import load_dataset
from pathlib import Path
from datasets import load_dataset, DatasetDict, Dataset
from typing import Final, List
from src.utils import setup_data_directory  


# Constants for maintainability
DATASET_NAME: Final[str] = "cfilt/iitb-english-hindi"
DEFAULT_OUTPUT_DIR: Final[str] = "data/raw"

def download_iitb_data(dataset_name: str = DATASET_NAME) -> None:
    """
    Downloads and prepares the IITB data.
    """
    data_result:bool = setup_data_directory()
    if not data_result: 
        print("Data already present.")
        return
    print(f"Downloading Dataset: {dataset_name}") 
    
    # Load the dataset from Hugging Face
    try: 
        ds: DatasetDict = load_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading dataset {dataset_name} from HF: {e}")
        return 

    for split in ds.keys():
        print(f"Processing split: {split}") 
        dataset: Dataset = ds[split] 
        translations: List[dict[str, str]] = dataset['translation']
        df:pd.DataFrame = pd.DataFrame({
            'en': [t['en'] for t in translations],
            'hi': [t['hi'] for t in translations]
        })
        save_path: Path = Path(DEFAULT_OUTPUT_DIR) / f"{split}.parquet"
        df.to_parquet(save_path, index=False, engine='pyarrow') 
        print(f"Saved {split} split to {save_path}")

    