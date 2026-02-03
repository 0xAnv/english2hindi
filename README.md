
###Proposed Directory Structure: 


transformer_iitb_project/
│
├── config/                 <-- The Control Room
│   └── config.py           (Hyperparameters: d_model=512, heads=8, etc.)
│
├── data/                   <-- The Fuel Station
│   ├── raw/                (Where you unzip the IITB dataset)
│   ├── processed/          (Cleaned text files ready for training)
│   └── tokenizer/          (Your saved Vocabulary/BPE tokenizer files)
│
├── src/                    <-- The Engine (The Core Logic)
│   ├── __init__.py
│   ├── dataset.py          (PyTorch Dataset & Dataloader logic)
│   ├── utils.py            (Helper functions: loading data, saving models)
│   │
│   └── model/              <-- The Transformer Architecture (Modular)
│       ├── __init__.py
│       ├── embeddings.py   (Input Embeddings + Positional Encoding)
│       ├── attention.py    (Multi-Head Attention mechanism)
│       ├── feed_forward.py (Position-wise Feed Forward Network)
│       ├── layers.py       (EncoderLayer and DecoderLayer blocks)
│       └── transformer.py  (Assembles the full Encoder-Decoder)
│
├── training/               <-- The Gym (Training Logic)
│   ├── trainer.py          (The training loop, validation loop)
│   ├── optimizer.py        (The Custom Learning Rate Scheduler from the paper)
│   └── loss.py             (CrossEntropy with Label Smoothing)
│
├── evaluation/             <-- The Report Card
│   ├── bleu_score.py       (Calculates BLEU score using sacrebleu/nltk)
│   └── translator.py       (Inference logic: Beam Search implementation)
│
├── notebooks/              <-- The Sandbox
│   └── 01_data_exploration.ipynb (Your "Virtual Lab" plays here)
│
├── train.py                <-- The "Start Button" script
└── evaluate.py             <-- The "Test Button" script