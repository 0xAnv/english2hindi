
##############################################################################
# logic to abstract data loaders etc 
# ############################################################################   

from abc import ABC, abstractmethod
from pathlib import Path 
from dataclasses import dataclass, field
from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd 

@dataclass
class BaseDataConfig:
    base_dir:Path 
    directories:list[str]
    format:str


class BaseDataLoader(ABC): 
    """Base class for loading datasets"""
    def __init__(self, config:BaseDataConfig) -> None:
        self.config:BaseDataConfig = config
        self._validate_paths()
    
    def _create_data_folder_structure(self, config:BaseDataConfig)-> None: 
        """creates the folder structures that hold datasets"""
        if not config.base_dir.exists(): 
            print(f"Creating data folder structure at {config.base_dir}")
            config.base_dir.mkdir(parents=True, exist_ok=True)
        
            for directory in config.directories:
                dir_path:Path = config.base_dir / directory
                dir_path.mkdir(parents=True, exist_ok=True)
        return

    
    def _validate_paths(self) -> None:
        """ Ensure all the files and folders are in place. if not create them."""
        if not self.config.base_dir.exists(): 
            self._create_data_folder_structure(self.config)
            print(f'Data files not found. Created folder structure at {self.config.base_dir}')
            print(f"Created base data directory at {self.config.base_dir}")
    
    # parquet file functions
    def _load_parquet_file(self, file_path:Path) -> pd.DataFrame:
        """Loads a parquet file into a pandas DataFrame"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return pd.read_parquet(file_path, engine='pyarrow')
    
    def _save_parquet_file(self, df:pd.DataFrame, file_path:Path) -> None:
        """Saves a pandas DataFrame to a parquet file"""
        assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
        assert isinstance(file_path, Path | str), "file_path must be a pathlib Path object or string"

        try: 
            df.to_parquet(file_path, index=False, engine='pyarrow')
        except Exception as e:
            raise ValueError(f"Failed to save parquet file {file_path}: {e}")
    
    @abstractmethod
    def load_raw_data(self, splits:list[str]) -> dict[str, pd.DataFrame]: 
        """
        Loads raw data files into pandas DataFrames
        Returns: 
            Dict[str, pd.DataFrame]: {'train': pd.DataFrame, 'test': pd.DataFrame, 'validation': pd.DataFrame}
        """
        pass

##############################################################################
# IITB Data Handler CODE
# ############################################################################  
@dataclass
class IITBDataConfig(BaseDataConfig):
    """Data configuration for IITB dataset"""
    # all other variables are same for this dataset
    base_dir:Path = Path("data/iitb")
    format:str = "parquet"
    directories:list[str] = field(default_factory=lambda: ['raw', 'processed', 'tokenised'])
    dataset_name:str = "cfilt/iitb-english-hindi"

class IITBDataLoader(BaseDataLoader):
    """Data Loader for IITB dataset"""
    def __init__(self, config:IITBDataConfig) -> None:
        super().__init__(config)
        if not config.base_dir.exists():
            print(f"Data directory {config.base_dir} does not exist. Downloading data...")
            self._download_iitb_data(config.dataset_name) # automatically downloads data 

   
    def _download_iitb_data(self, dataset_name: str) -> None: 
        """ Downloads IITB data"""
        # note: directories will exist automatically 
        # because our base class creates them upon initialization
        
        print(f"Downloading Dataset: {dataset_name}")
        # load the dataset from hugging face 
        try: 
            ds: DatasetDict = load_dataset(dataset_name)
        except Exception as e:
            raise ValueError(f"Failed to download dataset {dataset_name}: {e}")
            return
        
        for split in ds.keys():
            print(f"Processing split: {split}") 
            dataset: Dataset = ds[split] 
            translations: list[dict[str, str]] = dataset['translation']
            df:pd.DataFrame = pd.DataFrame({
                'en': [t['en'] for t in translations],
                'hi': [t['hi'] for t in translations]
            })
            save_path: Path = self.config.base_dir / 'raw' / f"{split}.{self.config.format}"
            self._save_parquet_file(df, save_path)
            print(f"Saved {split} split to {save_path}")

    def load_raw_data(self, splits:list[str] = ['train', 'test', 'validation']) -> dict[str, pd.DataFrame]:
        """Loads raw IITB data files into pandas DataFrames"""
        data_frames:dict[str, pd.DataFrame] = {}
        for split in splits:
            file_path:Path = self.config.base_dir / 'raw' / f"{split}.{self.config.format}"
            df:pd.DataFrame = self._load_parquet_file(Path(file_path))
            data_frames[split] = df
        return data_frames
    
def testing_iitb_data_loader() -> None:
    """ Function to test IITB data loader"""
    # Example usage
    iitb_config:IITBDataConfig = IITBDataConfig()
    iitb_loader:IITBDataLoader = IITBDataLoader(iitb_config)
    data:dict[str, pd.DataFrame] = iitb_loader.load_raw_data()
    for split, df in data.items():
        print(f"{split} data shape: {df.shape}")

    """
    Output of testing iitb loader :
    
    train data shape: (1659083, 2)
    test data shape: (2507, 2)
    validation data shape: (520, 2)
    """

    return None

if __name__ == "__main__":
    testing_iitb_data_loader()
    