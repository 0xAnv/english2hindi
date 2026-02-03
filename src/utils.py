"""
Helper functions for loading data, saving models etc 
"""

import yaml

from pathlib import Path
from typing import Final, List

# Define subfolders as a Final constant to prevent accidental modification
DATA_SUBFOLDERS: Final[List[str]] = ["raw", "processed", "tokenised"]

def setup_data_directory(root_path: str | Path = ".") -> bool:
    """
    Args:
        root_path (str | Path): root path 
    Returns:
        bool: True if the entire structure already existed.
              False if any folder (root or sub) had to be created.
    """

    root: Path = Path(root_path)
    data_dir: Path = root / "data"
    
    created_something: bool = False

    # 1. Handle main data directory
    if not data_dir.exists():
        print(f"Initializing directory: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)
        created_something = True

    # 2. Handle sub-hierarchy
    for sub in DATA_SUBFOLDERS:
        path: Path = data_dir / sub
        if not path.exists():
            print(f"Initializing subdirectory: {path}")
            path.mkdir(parents=True, exist_ok=True)
            created_something = True

    return created_something

def read_config(file_path: str | Path) -> dict:
    """
    Reads a YAML file and returns its contents "
    Args: 
        file_path (str | Path): path to the YAML file
    Returns:
        dict: contents of the YAML file
    """
    with open(file_path, 'r') as file:
        # safe_load handles standard YAML types safely
        data = yaml.safe_load(file)
    return data

def prepare_iitb_data():
    """
    Prepares the IITB data directory structure.
    """
    setup_data_directory()
    

setup_data_directory()