import os

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
import evaluate

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

def test_bleurt(path):
    from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

    df = pd.read_csv(path)
    
    # Get BLEURT scores
    checkpoint = f'lucadiliello/BLEURT-20-D12'

    bleurt = BleurtForSequenceClassification.from_pretrained(checkpoint) 
    device = torch.device('cpu')
    bleurt.to(device)
    bleurt.eval()
    tokenizer = BleurtTokenizer.from_pretrained(checkpoint)

    with torch.no_grad():
        inputs = tokenizer(df['pred'].tolist(), df['ref'].tolist(), padding='longest', return_tensors='pt').to(device)
        bleurt_scores = bleurt(**inputs).logits.flatten().cpu().tolist()
    
    # Pearson Corrlation
    pearson = pearsonr(bleurt_scores, df['score']).statistic
    print(f'Pearson correlation: {pearson}')

    # kendall tau-like
    kendall = get_concordant_discordant(bleurt_scores, df['score'])
    print(f'Kendall Tau-like: {kendall}')

def get_concordant_discordant(a, b):
    con = 0
    dis = 0
    for i in range(len(a)):
        for j in range(i, len(a)):
            if (a[j] - a[i]) * (b[j] - b[i]) > 0:
                con += 1
            else:
                dis += 1
    return (con - dis) / (con + dis)
    
def resolve_reference(src_path, processed_path):
    src_df = pd.read_csv(src_path)
    processed_df = pd.read_csv(processed_path)
    
    ref = []
    for i in range(len(processed_df)):
        src = processed_df.iloc[i]['src']
        for j in range(len(src_df)):
            if src_df.iloc[j]['original'] == src and src_df.iloc[j]['system'] == 'Human 1 Writing':
                ref.append(src_df.iloc[j]['generation'])
                break
    
    out = {
        'src': processed_df['src'].tolist(),
        'pred': processed_df['pred'].tolist(),
        'score': processed_df['score'].tolist(),
        'ref': ref
        }
    pd.DataFrame(out).to_csv(processed_path)

def resolve_reference_da(src_path, processed_path):
    src_df = pd.read_excel(src_path)
    processed_df = pd.read_csv(processed_path)
    
    ref = []
    for i in range(len(processed_df)):
        src = processed_df.iloc[i]['src']
        for j in range(len(src_df)):
            # Dirty fix
            if src[: 4] == 'Bone':
                ref.append('Bone has published books which include recipes by foragers, mycologists, and chefs that involve mushroom-based dishes.')
                break
            elif 'In a difficult situation' in src:
                ref.append('A hard situation encouraged him to study Graphic Design in 2007. Since then, he?€?s been a Cinematographer in the film industry. He also has a lot of experience with photography and graphic design.')
                break
            elif src_df.iloc[j]['Input.original'] == src and src_df.iloc[j]['Input.system'] == 'Human 1 Writing':
                ref.append(src_df.iloc[j]['Input.simplified'])
                break
        if len(ref) == i:
            print(src)
            raise
    
    out = {
        'src': processed_df['src'].tolist(),
        'pred': processed_df['pred'].tolist(),
        'score': processed_df['score'].tolist(),
        'ref': ref
        }
    pd.DataFrame(out).to_csv(processed_path)

def test_lens(path):
    import lens
    from lens.lens_score import LENS

    df = pd.read_csv(path)
    
    # Get BLEURT scores
    metric = LENS(uglobals.LENS_DIR, rescale=True)

    with torch.no_grad():
        lens_scores = metric.score(df['src'].tolist(), df['pred'].tolist(), [[ref] for ref in df['ref'].tolist()], batch_size=8, gpus=0)
    
    # Pearson Corrlation
    pearson = pearsonr(lens_scores, df['score']).statistic
    print(f'Pearson correlation: {pearson}')

    # kendall tau-like
    kendall = get_concordant_discordant(lens_scores, df['score'])
    print(f'Kendall Tau-like: {kendall}')

def test_sari(path):
    from easse.sari import corpus_sari

    df = pd.read_csv(path)
    
    scores = []
    for i in range(len(df)):
        src = df.iloc[i]['src']
        pred = df.iloc[i]['pred']
        ref = df.iloc[i]['ref']
        score = corpus_sari(orig_sents=[src], sys_sents=[pred], refs_sents=[[ref]])
        scores.append(score)
    
    # Pearson Corrlation
    pearson = pearsonr(scores, df['score']).statistic
    print(f'Pearson correlation: {pearson}')

    # kendall tau-like
    kendall = get_concordant_discordant(scores, df['score'])
    print(f'Kendall Tau-like: {kendall}')

def test_bleu(path):

    df = pd.read_csv(path)
    bleu = evaluate.load('bleu')
    
    scores = []
    for i in range(len(df)):
        src = df.iloc[i]['src']
        pred = df.iloc[i]['pred']
        ref = df.iloc[i]['ref']
        score = bleu.compute(predictions = [pred], references = [ref])['bleu']
        scores.append(score)
    
    # Pearson Corrlation
    pearson = pearsonr(scores, df['score']).statistic
    print(f'Pearson correlation: {pearson}')

    # kendall tau-like
    kendall = get_concordant_discordant(scores, df['score'])
    print(f'Kendall Tau-like: {kendall}')

def test_bertscore(path):

    df = pd.read_csv(path)
    bertscore = evaluate.load('bertscore')
    
    scores = []
    for i in range(len(df)):
        src = df.iloc[i]['src']
        pred = df.iloc[i]['pred']
        ref = df.iloc[i]['ref']
        score = bertscore.compute(predictions = [pred], references = [ref], lang='en')['f1'][0]
        scores.append(score)
    
    # Pearson Corrlation
    pearson = pearsonr(scores, df['score']).statistic
    print(f'Pearson correlation: {pearson}')

    # kendall tau-like
    kendall = get_concordant_discordant(scores, df['score'])
    print(f'Kendall Tau-like: {kendall}')

def test_FKGL(path):
    import textstat

    df = pd.read_csv(path)
    bertscore = evaluate.load('bertscore')
    
    scores = []
    for i in range(len(df)):
        src = df.iloc[i]['src']
        pred = df.iloc[i]['pred']
        ref = df.iloc[i]['ref']
        score = textstat.syllable_count(pred) / textstat.lexicon_count(pred)
        scores.append(score)
    
    # Pearson Corrlation
    pearson = pearsonr(scores, df['score']).statistic
    print(f'Pearson correlation: {pearson}')

    # kendall tau-like
    kendall = get_concordant_discordant(scores, df['score'])
    print(f'Kendall Tau-like: {kendall}')