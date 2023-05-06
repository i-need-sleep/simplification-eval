import argparse
import random

import pandas as pd
import numpy as np
import nltk

import utils.globals as uglobals

class Augment():
    def __init__(self):
        self.augment_ratio = 0.4 # augment_size/original_size

        self.aug_methods = [self.deletion, self.swap, self.scramble]
        self.aug_probs = [0.3, 0.3, 0.4] # For each method

        self.max_del_n_tokens = 4 # Uniformly sample from all integers <= max_del_n_tokens
        self.max_scramble_n_tokens = 5 # Uniformly sample from all integers <= max_del_n_tokens

        self.tokenizer = nltk.TreebankWordTokenizer()
        self.detoknizer = nltk.TreebankWordDetokenizer()

    def deletion(self, src, pred):
        # Randomly delete tokens. The number of deleted tokens is drawn uniformly from [1, min(sent_len - 1, self.max_del_n_tokens)]
        # Tokenize and determine how many tokens to delete
        tokenized = self.tokenizer.tokenize(pred)
        max_del_n_tokens = min(len(tokenized) - 1, self.max_del_n_tokens)
        if max_del_n_tokens == 0:
            return src, pred
        n_del = random.randint(1, max_del_n_tokens)

        # Select tokens to delete
        del_inds = np.random.choice([i for i in range(len(tokenized))], size=n_del, replace=False)
        tokens_out = []
        for token_idx, token in enumerate(tokenized):
            if token_idx not in del_inds:
                tokens_out.append(token)
        out = self.detoknizer.detokenize(tokens_out)
        return src, out
    
    def swap(self, src, pred):
        # Swap the complex and simplified sentences
        return pred, src
    
    def scramble(self, src, pred):
        # Randomly swap n pairs of tokens. 
        tokenized = self.tokenizer.tokenize(pred)
        max_scramble_n_tokens = min(len(tokenized) - 1, self.max_scramble_n_tokens)
        if max_scramble_n_tokens == 0:
            return src, pred
        n_scramble = random.randint(1, max_scramble_n_tokens)

        indices = [i for i in range(len(tokenized))]
        for _ in range(n_scramble):
            shuffle_inds = np.random.choice(indices, size=2, replace=False)
            keep = indices[shuffle_inds[0]]
            indices[shuffle_inds[0]] = indices[shuffle_inds[1]]
            indices[shuffle_inds[1]] = keep
            
        tokens_out = []
        for idx in indices:
            tokens_out.append(tokenized[idx])
        out = self.detoknizer.detokenize(tokens_out)
        return src, out
    
    def apply_augmentation(self, src, pred):
        pred = str(pred).strip()
        # Is this sentence selected for augmentation?
        if random.random() > self.augment_ratio or len(pred) == 0:
            return None, None
        
        augment_method_idx = np.random.choice([i for i in range(len(self.aug_methods))], p=self.aug_probs)
        augment_method = self.aug_methods[augment_method_idx]

        out = augment_method(src, pred)
        return out

def augment_preds(augment, paths):
    
    # Concat into [src, ...], [pred, ...]
    for path_idx, path in enumerate(paths):
        print(path)
        df = pd.read_csv(path)
        if path_idx == 0:
            srcs = df['src'].tolist()
            preds = df['pred'].tolist()
        else:
            srcs = srcs + df['src'].tolist()
            preds = preds + df['pred'].tolist()

    # Apply augmentation
    srcs_out, augmented_out = [], []
    for idx, (src, pred) in enumerate(zip(srcs, preds)):
        src, augmented = augment.apply_augmentation(src, pred)
        if augmented == None:
            continue
        srcs_out.append(src)
        augmented_out.append(augmented)

    df = pd.DataFrame({
        'src': srcs_out,
        'pred': augmented_out 
    })

    save_path = f'{uglobals.PROCESSED_DIR}/openwebtext/augmented.csv'
    df.to_csv(save_path)
    print(f'Saved at {save_path}')
    return

if __name__ == '__main__':
    paths = [f'{uglobals.PROCESSED_DIR}/openwebtext/gpt_turbo.csv', f'{uglobals.PROCESSED_DIR}/openwebtext/gpt_curie.csv']
    augment = Augment()
    augment_preds(augment, paths)