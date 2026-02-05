import pandas as pd 
from src.dataloader import IITBDataLoader, IITBDataConfig

# testing the iitb data loader 
def testing_iitb_data_loader() -> None:
    """ Function to test IITB data loader"""
    
    iitb_config:IITBDataConfig = IITBDataConfig()
    print(f"Testing IITB Data Loader with config: {iitb_config}")
    iitb_loader:IITBDataLoader = IITBDataLoader(iitb_config)
    print("IITB Data Loader initialized successfully.")
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
    print("Testing IITB Data Loader...")
    testing_iitb_data_loader()