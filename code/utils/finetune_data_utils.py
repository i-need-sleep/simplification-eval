import os

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

import utils.globals as uglobals

def simpeval_asset_make_splits(dev_size=10):
    df = pd.read_csv(f'{uglobals.STAGE3_DIR}/simpeval_asset.csv')
    
    # Select original sentences for the dev set
    dev_indices = np.random.choice([i for i in range(100)], size = dev_size, replace = False).tolist()
    
    train = [[], [], []] # src, pred, score
    dev = [[], [], []]
    for i in range(len(df)):
        line = df.iloc[i]

        if line['original_id'] in dev_indices:
            lis = dev
        else:
            lis = train
        
        lis[0].append(line['original'])
        lis[1].append(line['generation'])

        scores = [line[f'rating_{str(i+1)}_z_score'] for i in range(5)]
        lis[2].append(sum(scores) / len(scores))

    train_out = {
        'src': train[0],
        'pred': train[1],
        'score': train[2],
    }
    dev_out = {
        'src': dev[0],
        'pred': dev[1],
        'score': dev[2],
    }
    pd.DataFrame(train_out).to_csv(f'{uglobals.STAGE3_PROCESSED_DIR}/simpeval_asset_train.csv', index=False)
    pd.DataFrame(dev_out).to_csv(f'{uglobals.STAGE3_PROCESSED_DIR}/simpeval_asset_dev.csv', index=False)

def process_simpeval_2022():
    df = pd.read_csv(f'{uglobals.STAGE3_DIR}/simpeval_2022.csv')
    
    test = [[], [], []]
    for i in range(len(df)):
        line = df.iloc[i]
        
        test[0].append(line['original'])
        test[1].append(line['generation'])

        scores = [line[f'rating_{str(i+1)}_zscore'] for i in range(3)]
        test[2].append(sum(scores) / len(scores))

    out = {
        'src': test[0],
        'pred': test[1],
        'score': test[2],
    }
    pd.DataFrame(out).to_csv(f'{uglobals.STAGE3_PROCESSED_DIR}/simpeval_2022.csv', index=False)

def process_simp_da(dev_size=5, test_size=25):
    df = pd.read_excel(f'{uglobals.STAGE3_DIR}/simpDA_2022.xlsx')
    
    # Normalize
    df.iloc[:, 5: ] = df.iloc[:, 5:].apply(lambda x: (x-x.mean())/ x.std(), axis=0)

    # Select original sentences for the dev/test splits
    dev_indices = np.random.choice([i for i in range(60)], size = dev_size, replace = False).tolist()
    indices = []
    for i in range(60):
        if i not in dev_indices:
            indices.append(i)
    test_indices = np.random.choice(indices, size = test_size, replace = False).tolist()

    train = [[] for _ in range(5)]
    dev = [[] for _ in range(5)]
    test = [[] for _ in range(5)]
    for i in range(int(len(df) / 3)):
        line = df.iloc[3 * i]
        lines = df.iloc[3 * i: 3 * i + 3]
        adequacy = np.average(np.array(lines.iloc[:, -3]))
        fluency = np.average(np.array(lines.iloc[:, -2]))
        simplicity = np.average(np.array(lines.iloc[:, -1]))

        lis = train
        if line['Input.id'] in dev_indices:
            lis = dev
        elif line['Input.id'] in test_indices:
            lis = test

        lis[0].append(line['Input.original'])
        lis[1].append(line['Input.simplified'])
        lis[2].append(adequacy)
        lis[3].append(fluency)
        lis[4].append(simplicity)
    
    def save_splits(lis, name):
        for idx, score_name in enumerate(['adquacy', 'fluency', 'simplicity']):
            out = {
                'src': lis[0],
                'pred': lis[1],
                'score': lis[2 + idx]
            }
            pd.DataFrame(out).to_csv(f'{uglobals.STAGE3_PROCESSED_DIR}/simp_da_{name}_{score_name}.csv')

    save_splits(train, 'train')
    save_splits(dev, 'dev')
    save_splits(test, 'test')

class FinetuneDataset(Dataset):
    def __init__(self, aggregated_path, tokenizer):
        self.n_supervision = 13 # The total of the supervision signals (the number of regression heads) including the ones to be filled as 0

        self.df = pd.read_csv(aggregated_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df) 

    def __getitem__(self, i):
        line = self.df.iloc[i]
        
        # Concat src and pred 
        sent = str(line['src']) + ' ' + self.tokenizer.sep_token + ' ' + str(line['pred'])

        out = {
            'sent': sent,
            'scores': [0 for _ in range(self.n_supervision - 1)] + [line['score']],
            'score_mask': [0 for _ in range(self.n_supervision - 1)] + [1],
        }
        return out

def mr_collate(batch):
    for idx, line in enumerate(batch):
        if idx == 0:
            sent = [line['sent']]
            scores = torch.tensor(line['scores']).unsqueeze(0)
            score_mask = torch.tensor(line['score_mask']).unsqueeze(0)
        else:
            sent.append(line['sent'])
            scores = torch.cat((scores, torch.tensor(line['scores']).unsqueeze(0)), dim=0).float()
            score_mask = torch.cat((score_mask, torch.tensor(line['score_mask']).unsqueeze(0)), dim=0).float()
    return sent, scores, score_mask
    
def make_finetuning_loader(path, tokenizer, batch_size, shuffle=True):
    dataset = FinetuneDataset(path, tokenizer)
    print(f'Making dataloader: {path}')
    print(f'# samples: {len(dataset)}')
    loader = DataLoader(dataset, batch_size=batch_size ,shuffle=shuffle, collate_fn=mr_collate)
    return loader

def test_bleurt_simpeval_2022():
    df = pd.read_csv(f'{uglobals.STAGE3_DIR}/simpeval_2022.csv')
    processed_df = pd.read_csv(f'{uglobals.STAGE3_PROCESSED_DIR}/simpeval_2022.csv')

    preds = processed_df['pred'].tolist()
    refs  = []
    for i in range(len(df)):
        line = df.iloc[i]
        print(line['system'])
        if line['system'] == 'asset.test.simp':
            refs.append(line['generation'])
    
    print(len(preds))
    print(len(refs))
    exit()
    
    # Get BLEURT scores
    checkpoint = f'lucadiliello/{checkpoint}'

    bleurt = BleurtForSequenceClassification.from_pretrained(checkpoint) 
    device = torch.device('cpu')
    bleurt.to(device)
    bleurt.eval()
    tokenizer = BleurtTokenizer.from_pretrained(checkpoint)

    out = [] # [score, ...]

    with torch.no_grad():
        inputs = tokenizer(text_inputs, [self.ref for _ in text_inputs], padding='longest', return_tensors='pt').to(self.device)
        out = bleurt(**inputs).logits.flatten().cpu()
        out = out.tolist()
    