import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        texts = [subtext.lower().split() for subtext in texts]
        texts_list = list(set([element for innerList in texts for element in innerList]))
        texts_list = sorted(texts_list)
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3
        for i, word in enumerate(texts_list):
            self.word_to_id[word] = 4+i
        self.vocab_size = len(self.word_to_id)
        self.id_to_word = {v : k for k, v in self.word_to_id.items()}
        print(self.vocab_size)
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        words = text.lower().split()
        words = sorted(words)
        encoded = []
        for word in words:
            if word in self.word_to_id.keys(): 
                encoded.append(self.word_to_id[word])
            else:
                encoded.append(self.word_to_id[self.unk_token])

        return encoded
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        print(self.id_to_word)
        print(ids)
        decoded = []
        for i in ids:
            if i in self.id_to_word.keys():   
                decoded.append(self.id_to_word[i])
            else:
                decoded.append(self.id_to_word[1])

        return " ".join(decoded)

