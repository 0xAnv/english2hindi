#####################################################
# Logic for tokenisation of input strings into tokens.
#####################################################

# module imports 
import os
import json
from abc import abstractmethod, ABC 
from pathlib import Path

# bpe specific imports 
from tokenizers import Tokenizer as RustTokenizer
from tokenizers import models, pre_tokenizers, trainers
import tokenizers


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

    
#####################################################################
# Implementation of Byte Pair Encoding (BPE) Tokeniser
#####################################################################
"""
"""

class BPETokenizer(Tokenizer): 
    def __init__(self) -> None:
        super().__init__()
        self._tokenizer : RustTokenizer = RustTokenizer(models.BPE(unk_token="[UNK]"))
        self._tokenizer.pre_tokenizer  = pre_tokenizers.Whitespace() # you cannot type annotate an attribute assignment ( in this _tokenizer.case pre_tokenizer)

    def train(self, text_data:list[str], vocab_size:int) -> None :
        """Trains the BPE tokeniser on a list of strings (sentences) """
        trainer : trainers.BpeTrainer = trainers.BpeTrainer(
            vocab_size=vocab_size, 
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"]
        )
        
        # running the training 
        # We pass the data (iterator) and the trainer to the engine.
        # It runs very fast (Rust speed).
        self._tokenizer.train_from_iterator(text_data, trainer=trainer)
        # Now that Rust is done, we copy the learned vocab back to Python
        # so your Base Class properties work correctly.
        self.vocab = self._tokenizer.get_vocab()
        if self.vocab is None:
            print("Training failed, vocab is NoneType object")
            return
        
        # acknowledge training completion
        print("training completed. Vocab size:", len(self.vocab))

    # Encoding and decoding functions for bpe goes here 
    def encode(self, text:str) -> list[int] : 
        """ Input: single string "hello i am anvesh"
            Output: List of integers [23, 44, 90, 54]
        """
        encoded  = self._tokenizer.encode(text)
        return encoded.ids
    
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    
    # Loading and saving a trained tokenizer goes here 
    def save(self, directory : str | Path, prefix : str) -> None:
        """ Saves the BPE tokeniser using Rust Tokenizer's save method"""
        dir_path : Path = Path(directory) 
        dir_path.mkdir(parents=True , exist_ok=True)
        
        # save full model merges + vocab 
        model_path: Path = dir_path / f"{prefix}_tokenizer.json"
        print(f"Saving model to path: {model_path}")
        return None 
    
    def load(self, directory: str | Path, prefix:str) -> None:
        """Loads the BPE tokeniser using Rust Tokenizer's load method"""
        dir_path: Path = Path(directory) 
        model_path: Path = dir_path / f"{prefix}_tokenizer.json"
        if not model_path.exists(): 
            raise FileNotFoundError(f"Tokeniser file {model_path} DOESN'T EXIST ")
        print(f"Loading model from path: {model_path}")

        # load rust backend 
        self._tokenizer : RustTokenizer = RustTokenizer.from_file(str(model_path))
        
        # updating base class state so vocab_size works 
        self.vocab = self._tokenizer.get_vocab()

        print("Loaded vocab size:", len(self.vocab) if self.vocab else 0)
        return None
    

    