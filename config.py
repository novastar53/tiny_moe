from dataclasses import dataclass


@dataclass
class Config:
    vocab_size: int = 49152
    block_size: int = 128
    n_layer: int = 8 
    n_embed: int = 64 
    n_hidden: int = 64*3
    n_head: int = 8

