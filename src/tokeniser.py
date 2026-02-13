# abstract base class imports 
from abc import ABC, abstractmethod

# HF tokeniser specifics
from numpy import iterable
from tokenizers import Tokenizer 
from tokenizers import models
from tokenizers import pre_tokenizers
from tokenizers import trainers
from tokenizers import decoders

# other imports
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Iterable

from src.dataloader import BaseDataLoader, IITBDataLoader, IITBDataConfig
###########################################################
# Base class for tokeniser
###########################################################
 
class Tokeniser(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.vocab : dict | None = None # this is our none
    
    # training the tokeniser
    @abstractmethod
    def train(self) -> None: pass 

    # encoding and decoding logic go here 
    @abstractmethod
    def encode(
        self, 
        data: list[str] | Iterable[list[str]]
    ) -> list[list[int]] | Iterable[list[list[int]]]: pass

    @abstractmethod
    def decode(
        self, 
        tokens: list[list[int]] | Iterable[list[list[int]]]
    ) -> list[str] | Iterable[list[str]] : pass

    # IO related functions go here 
    @abstractmethod
    def save(self) -> None : pass 
    @abstractmethod
    def load(self) -> None : pass

@dataclass
class BPEConfig:
    vocab_size:int #vocabulary size total
    unk_token: str
    end_of_word_suffix: str 
    special_tokens: list[str]
    text_data: list[str] | Iterable[list[str]] # this is the text data that we will use to train the tokeniser
    
    # tokeniser saving configs
    save_name:str 
    save_path:Path


class BPETokeniser(Tokeniser):
    def __init__(self, config: BPEConfig) -> None:
        super().__init__()
        self.config = config
        # creating hf tokeniser object
        self._tokenizer: Tokenizer = Tokenizer(models.BPE(
            unk_token= self.config.unk_token, 
            end_of_word_suffix= self.config.end_of_word_suffix
        ))
        # this is a whitespace pre tokenizer
        self._tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        # automatically train the tokeniser on initialisation 
        self.tokeniser_save_path:Path = Path(self.config.save_path)

        if not self.tokeniser_save_path.exists(): 
            print("No folder found to save tokeniser... making one")
            self.tokeniser_save_path.mkdir(exist_ok=True)

        if Path(self.tokeniser_save_path, self.config.save_name).exists() : 
            self.load() # read from config
        
        else: self.train()

    def train(self) -> None : 
        trainer: trainers.BpeTrainer = trainers.BpeTrainer(
            vocab_size = self.config.vocab_size , 
            end_of_word_suffix = self.config.end_of_word_suffix, 
            special_tokens = self.config.special_tokens
        )
        print("Tokeniser training started...")
        self._tokenizer.train_from_iterator(
            iterator = self.config.text_data, 
            trainer = trainer
        )

        print("Tokeniser training completed") 
        self.vocab = self._tokenizer.get_vocab()
        if self.vocab is None:
            print("Something is wrong, vocab is NoneType object")
            return
        self.save()
        return None

    # encode decode logical functions (depends on the tokeniser)
    def encode(
            self, 
            data: list[str] | Iterable[list[str]]
            ) -> list[list[int]] | Iterable[list[list[int]]]:
        # i was too lazy to code iterable logic. I will write it when i need it. FOr now, loading entire data into memory is not a problem. (i've got 32 gb of ram :P)

        
        assert self.vocab is not None, "Vocab is not trained yet. Please train the tokeniser before encoding."
        assert isinstance(data, list) or isinstance(data, Iterable), "Input data must be a list of strings or an iterable of list of strings"

        if isinstance(data, list) and isinstance(data[0], str): 
            # this is a list of strings, we can encode it directly
            encoded : list[list[int]] = [self._tokenizer.encode(text).ids for text in data]
            return encoded
        raise ValueError("Input data must be a list of strings or an iterable of list of strings")

    def decode(
            self, 
            tokens: list[list[int]] | Iterable[list[list[int]]] 
            ) -> list[str] | Iterable[list[str]]:
        
        self._tokenizer.decoder = decoders.BPEDecoder(
            suffix = self.config.end_of_word_suffix
        )

        if isinstance(tokens, list) and isinstance(tokens[0], list) and isinstance(tokens[0][0], int):
            decoded_str : list[str] = [self._tokenizer.decode(tokens_list) for tokens_list in tokens]
            return decoded_str
        
        else : raise ValueError("please send list[list[int]] datatype")
        

    # IO Functions of the class 
    def load(self) -> None :
        filepath:Path = self.tokeniser_save_path / self.config.save_name 
        self._tokenizer = Tokenizer.from_file(str(filepath))
        print(f"Read file path: {filepath}")
        self.vocab = self._tokenizer.get_vocab()
        print(f"Loaded tokeniser from {filepath}")
        return None
              
    def save(self) -> None : 
        print(f"Saving the tokeniser in {self.tokeniser_save_path}")
        filename:str = self.config.save_name 
        self._tokenizer.save(str(self.tokeniser_save_path / filename), pretty=True)


def test_bpe_tokeniser() -> None:
    # creating some dummy data for testing 
    text_data = [
        "Hello world",
        "This is a test",
        "Tokeniser is working"
    ]

    config = BPEConfig(
        vocab_size=10,
        unk_token="[UNK]",
        end_of_word_suffix="</w>",
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
        text_data=text_data,

        save_name="bpe.json", 
        save_path= Path("./tokeniser")
    )

    tokeniser: BPETokeniser = BPETokeniser(config=config)
    print(tokeniser.vocab)
    print("-"*60)
    encodings:list[list[int]] | Iterable[list[list[int]]] = tokeniser.encode(data = config.text_data)
    print(encodings)

    print("-"*60)
    decodings: list[str] | Iterable[list[str]] = tokeniser.decode(tokens=encodings)
    print(decodings)

#########################################################
# Tokenisation pipelining code
#########################################################
import pandas as pd
from src.dataloader import IITBDataConfig, IITBDataLoader


# Stage 1: Loading Data and making a combined object 

# creating Data loaders & config files 
dataconf: IITBDataConfig = IITBDataConfig(
    base_dir= Path("datasets/iitb"), 
    format="parquet", 
    directories=["raw", "processed", "tokenised"] , 
    dataset_name= "cfilt/iitb-english-hindi"
)

dataloader: IITBDataLoader = IITBDataLoader(config=dataconf)

print("Testing DataLoader") 

# data is loaded and converted to a combined python list 
data = dataloader.load_raw_data(splits=["train"])['train']
eng_sentences:list[str] = data['en'].to_list()
hi_sentences : list[str] = data['hi'].to_list()
entire_corpus : list[str]= eng_sentences + hi_sentences

print("*"*80)

print("stage 1 succesfully completed. Data has been merged into a single python list iterable object")


# stage 2 : training a tokeniser on this entire corpurse and testing the tokeniser out 

tokenconf : BPEConfig = BPEConfig(
    vocab_size=32000 , 
    unk_token= "[UNK]" , 
    end_of_word_suffix= "</w>", 
    special_tokens=["[UNK]", "[PAD]", "[SOS]","[EOS]"],  
    save_name= "bpe_tokeniser_32k.json", 
    save_path= dataconf.base_dir / Path("tokeniser"), 

    # text data came from dataloader pipeline
    text_data= entire_corpus
)

tokeniser : BPETokeniser = BPETokeniser(config=tokenconf)
print("Tokeniser is trained and loaded into memory. Ready for encoding and decoding...")
test_sent:str = "My name is Anvesh Dange"
encoded = tokeniser.encode([test_sent])
decoded = tokeniser.decode(encoded)
print(f"{test_sent} ----> {encoded}") 
print(f"{encoded} ----> {decoded}")
print("*"*80)
print("stage 2 succesfully completed")

tokenised_file_path: Path = dataconf.base_dir / Path("tokenised/train_tokens.parquet") 
if not Path(dataconf.base_dir, "tokenised", "train_tokens.parquet").exists():
    print("Creating new dataframes for tokenized version of our data")
    tokenised_df : pd.DataFrame = data
    print("Tokenising ALL english sentences (will take time)")
    # tokenised_english_sentences: list[list[int]] | Iterable[list[list[int]]] = tokeniser.encode(data = eng_sentences)
    tokenised_df['en'] = tokeniser.encode(tokenised_df['en'].to_list())

    print("Tokenising ALL hindi sentences (will take time)")
    # tokenised_hindi_sentences: list[list[int]] | Iterable[list[list[int]]] = tokeniser.encode(data=hi_sentences)
    tokenised_df['hi'] = tokeniser.encode(tokenised_df['hi'].to_list())

    print("succesfully tokenised English and Hindi sentences")
     
    tokenised_df.to_parquet(path= str(tokenised_file_path), index=False, engine="pyarrow")

print("*"*80)
print("stage 3 succesfully completed")
tokenised_df = pd.read_parquet(path=tokenised_file_path, engine="pyarrow")
for i in range(10): 
    eng = tokenised_df.iloc[i]['en'].tolist()
    hi = tokenised_df.iloc[i]['hi'].tolist()
    print(f"{tokeniser.decode([eng])} ----> {tokeniser.decode([hi])}")



print(f"Successfully saved Tokenised file to : {tokenised_file_path} ")