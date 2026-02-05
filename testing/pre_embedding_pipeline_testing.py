import pandas as pd
from src.dataloader import IITBDataConfig, IITBDataLoader
from src.tokeniser import BPETokenizer

def get_iitb_data(datapoints:int = 1000) -> pd.DataFrame: 
    configf : IITBDataConfig = IITBDataConfig()
    loader : IITBDataLoader = IITBDataLoader(configf)
    data : dict[str, pd.DataFrame] = loader.load_raw_data(splits=["train"])
    d : pd.DataFrame = data['train']
    d = d.head(datapoints)
    return d

def tokenize(data:pd.DataFrame) -> list[list[int]]:
    bpe: BPETokenizer = BPETokenizer()
    sentences: list[str] = data['en'].tolist()
    bpe.train(sentences, vocab_size=1000)
    output: list[list[int]] = []
    for sentence in sentences:
        tokenized: list[int] = bpe.encode(sentence)
        output.append(tokenized)
    print("Tokenizer Vocabulary")
    print(bpe.vocab)
    return output

def get_vocab(tokenizer:BPETokenizer) -> dict[str, int]:
    return tokenizer.vocab if tokenizer.vocab is not None else {}

if __name__ == "__main__":
    data = get_iitb_data(datapoints=2)
    tokenized = tokenize(data)

    print("Pre-embedding pipeline testing complete.")
    print("Outputs:  --------------")
    print("Input Sentences: ")
    for sentence in data['en'].tolist():
        print(sentence)
    print("Tokenized Outputs: ")
    for tokens in tokenized:
        print(tokens)
    print("Vocab: ")