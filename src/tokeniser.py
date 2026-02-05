#####################################################
# Logic for tokenisation of input strings into tokens.
#####################################################

# module imports 
import os
import json

from abc import abstractmethod, ABC 
from pathlib import Path

# tokeniser base class
class Tokenizer(ABC): 
    """Generic class for any tokeniser. They will inherit this class"""
    def __init__(self) -> None:
        super().__init__()
        # every tokeniser will need a vocabulary size 
        # we initialise it as empty 
        # { 'apple' : 3, 'potato' : 2 }
        self.vocab : dict[str,int] | None = None # map from Token -> ID 
        self.id_to_token : dict[int, str] | None = None # map from ID -> Token

    @abstractmethod
    def train(self, text_data:list[str], vocab_size:int) -> None : 
        """
        Trains the tokeniser on a list of strings (sentences) 
        This function will return nothing, but it will update internal state.
        """
        pass 

    @abstractmethod
    def encode(self, text:str) -> list[int] : 
        """
        Input: single string "hello i am anvesh"
        Output: List of integers [23, 44, 90, 54]
        """
        pass 

    @abstractmethod
    def decode(self, ids:list[int]) -> str:
        """
        Input: List of tokens
        Output: A reconstructed string
        """
        pass

    # when we load the pipeline, we will have to expose the vocab size property from internal state 
    # this will be used in later part of training pipeline like the nn.EmbeddingLayer()
    @property
    def vocab_size(self) -> int : 
        if self.vocab is None: return 0
        return len(self.vocab)
    
    # Generic Functions to save and load the tokeniser
    def save(self, directory : str | Path, prefix : str) -> None:
        """ Saves the vocab into a single json file"""
        if self.vocab is None:
            print("Don't be rookie. First train your tokeniser")
            return
        
        dir_path : Path = Path(directory)
        dir_path.mkdir(parents=True , exist_ok=True)

        vocab_path:Path = dir_path / f"{prefix}_vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        return None
    
    def load(self, directory: str | Path, prefix:str) -> None:
        """Loads the vocab from json file using pathlib"""
        dir_path: Path = Path(directory) 
        vocab_file_path: Path = dir_path / f"{prefix}_vocab.json"

        if not vocab_file_path.exists(): raise FileNotFoundError(f"Tokeniser file {vocab_file_path} DOESN'T EXIST ")
        with open(vocab_file_path, "r", encoding="utf-8") as f: 
            print("Loaded Vocab in self.vocab")
            self.vocab = json.load(f)
        
        # automatically generate the reverse mapping for all child classes
        if self.vocab is None: 
            print("Cannot create reverse mapping, VOCAB is NoneType Object")
            return
        # creating reverse mapping
        self.id_to_token = {v:k for k, v in self.vocab.items()}

    