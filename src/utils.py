"""Utility functions"""
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers import decoders

def tokeniser_from_file(filepath: Path, padid:int, padtok:str, max_seq_len:int) -> Tokenizer : 
    """returns a BPE tokeniser object from trained """
    if not filepath.exists(): raise FileNotFoundError(f"File not found at : {filepath}")
    tokeniser:Tokenizer = Tokenizer.from_file(str(filepath))
    tokeniser.decoder = decoders.BPEDecoder(suffix="</w>")
    tokeniser.enable_padding(pad_id=padid, pad_token=padtok,length=max_seq_len)
    tokeniser.enable_truncation(max_length=max_seq_len)
    return tokeniser

def test_tokeniser_from_file(filepath: Path, padid:int, padtok:str, max_seq_len:int) -> None : 
    tokeniser = tokeniser_from_file(filepath=filepath, padid=padid, padtok=padtok, max_seq_len=max_seq_len)
    sentence:str = "[SOS]I am anvesh[EOS]"
    enc = tokeniser.encode(sentence)
    dec = tokeniser.decode(enc.ids)

    if(dec == "I am anvesh"): print(f"tokeniser_from_file() function passed test")

def tokeniser_from_scratch(): raise NotImplementedError

if __name__ == "__main__":
    test_tokeniser_from_file(
        filepath=Path("datasets/iitb/tokeniser/bpe_tokeniser_32k.json"), 
        padid=1,
        padtok='[PAD]',
        max_seq_len=128
    )
