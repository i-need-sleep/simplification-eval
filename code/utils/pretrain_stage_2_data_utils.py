import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import utils.globals as uglobals

def aggregate_raw():
    srcs = []
    refs = []
    for dataset_name in os.listdir(uglobals.DRESS_DIR):
        if os.path.isdir(f'{uglobals.DRESS_DIR}/{dataset_name}'):
            with open(f'{uglobals.DRESS_DIR}/{dataset_name}/test/Complex', 'r') as f:
                srcs += f.readlines()
            with open(f'{uglobals.DRESS_DIR}/{dataset_name}/test/Reference', 'r') as f:
                refs += f.readlines()
    df = pd.DataFrame({
        'src': srcs,
        'ref': refs
    })
    df.to_csv(f'{uglobals.STAGE2_RAW}/aggregated.csv', index=False)
    with open(f'{uglobals.STAGE2_RAW}/stage2_raw.en', 'w') as f:
        f.writelines(srcs)

def csv_add_ref(pred_path, ref_path):
    pred_df = pd.read_csv(pred_path)
    ref_df = pd.read_csv(ref_path)
    out_path = pred_path.replace('.csv', '_ref.csv')

    try:
        out = {
            'src': ref_df['src'].tolist(),
            'pred': pred_df['pred'].tolist(),
            'ref': ref_df['ref'].tolist()
        }
        out_df = pd.DataFrame(out)
    except:
        # Handle missing entries
        pred_srcs = pred_df['src'].tolist()
        preds = pred_df['pred'].tolist()
        ref_srcs = ref_df['src'].tolist()
        refs = ref_df['ref'].tolist()

        ref_out = []
        preds_out = []
        srcs_out = []
        for idx, pred_src in enumerate(pred_srcs):
            for ref_idx, ref_src in enumerate(ref_srcs):
                if pred_src == ref_src[:-1].strip():
                    ref_out.append(refs[ref_idx])
                    srcs_out.append(ref_src)
                    preds_out.append(preds[idx])
                    break
        out = {
            'src': srcs_out,
            'pred': preds_out,
            'ref': ref_out
        }
        out_df = pd.DataFrame(out)
    out_df.to_csv(out_path)

def model_preds_to_csv(src_path, pred_path, ref_path, out_path):
    with open(src_path, 'r', encoding='utf-8') as f:
        src = f.readlines()
    with open(pred_path, 'r', encoding='utf-8') as f:
        pred = f.readlines()
    with open(ref_path, 'r', encoding='utf-8') as f:
        ref = f.readlines()
    pd.DataFrame({
        'src': src,
        'pred': pred,
        'ref': ref
    }).to_csv(out_path)

def editnts_to_csv():
    for dataset in ['Newsela', 'WikiSmall', 'WikiLarge']:
        src_path = f'{uglobals.STAGE2_OUTPUTS_DIR}/dress/{dataset}/test/Complex'
        ref_path = f'{uglobals.STAGE2_OUTPUTS_DIR}/dress/{dataset}/test/Reference'
        pred_path = f'{uglobals.STAGE2_OUTPUTS_DIR}/editnts_test/{dataset.lower()}_editnts.txt'
        out_path = f'{uglobals.STAGE2_OUTPUTS_DIR}/referenced/editnts_{dataset}.csv'
        model_preds_to_csv(src_path, pred_path, ref_path, out_path)

def dress_to_csv():
    for dataset in ['Newsela', 'WikiSmall', 'WikiLarge']:
        src_path = f'{uglobals.STAGE2_OUTPUTS_DIR}/dress/{dataset}/test/Complex'
        ref_path = f'{uglobals.STAGE2_OUTPUTS_DIR}/dress/{dataset}/test/Reference'

        for model_name in os.listdir(f'{uglobals.STAGE2_OUTPUTS_DIR}/dress/{dataset}/test'):
            if model_name not in ['Complex', 'Reference', 'lower']:
                pred_path = f'{uglobals.STAGE2_OUTPUTS_DIR}/dress/{dataset}/test/{model_name}'
                out_path = f'{uglobals.STAGE2_OUTPUTS_DIR}/referenced/{model_name}_{dataset}.csv'
                model_preds_to_csv(src_path, pred_path, ref_path, out_path)

def aggregated_ref_csvs():
    dfs = []
    for file_name in os.listdir(f'{uglobals.STAGE2_OUTPUTS_DIR}/referenced'):
        dfs.append(pd.read_csv(f'{uglobals.STAGE2_OUTPUTS_DIR}/referenced/{file_name}'))
    df = pd.concat(dfs)
    df.to_csv(f'{uglobals.STAGE2_OUTPUTS_DIR}/aggregated.csv', index=False)

def stage2_aggregate_csvs(ref_free_path, ref_based_path, commonlit_src_path, commonlit_pred_path, out_path):
    ref_free_df = pd.read_csv(ref_free_path)
    ref_based_df = pd.read_csv(ref_based_path)
    commonlit_src_df = pd.read_csv(commonlit_src_path)
    commonlit_pred_df = pd.read_csv(commonlit_pred_path)

    out = {
        'src': ref_free_df['src'].tolist(),
        'pred': ref_free_df['pred'].tolist(),
        'ref': ref_based_df['ref'].tolist(),
        'self_bleu': ref_free_df['self_bleu'].tolist(),
        'self_bertscore': ref_free_df['self_bertscore'].tolist(),
        'sbert': ref_free_df['sbert'].tolist(),
        'src_perplexity': ref_free_df['src_perplexity'].tolist(),
        'pred_perplexity': ref_free_df['pred_perplexity'].tolist(),
        'src_syllable_per_word': ref_free_df['src_syllable_per_word'].tolist(),
        'pred_syllable_per_word': ref_free_df['pred_syllable_per_word'].tolist(),
        'commonlit_src': commonlit_src_df['target'].tolist(),
        'commonlit_pred': commonlit_pred_df['target'].tolist(),
        'bleu': ref_based_df['bleu'].tolist(),
        'bertscore': ref_based_df['bertscore'].tolist(),
        'sari': ref_based_df['sari'].tolist(),
    }
    pd.DataFrame(out).to_csv(out_path, index=False)
    return

def make_splits(path, dev_ratio=0.1):
    df = pd.read_csv(path)

    # Normalize
    df.iloc[:, 3: ] = df.iloc[:, 3:].apply(lambda x: (x-x.mean())/ x.std(), axis=0)

    # Split train/dev sets and save 
    df = df.sample(frac=1)
    dev_idx = int(round(len(df) * dev_ratio))
    
    dev_df = df.iloc[: dev_idx]
    train_df = df.iloc[dev_idx: ]

    dev_df.to_csv(path.replace('.csv', '_dev.csv'), index=False)
    train_df.to_csv(path.replace('.csv', '_train.csv'), index=False)
    return

class PretrainingStage2Dataset(Dataset):
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
            'scores': [
                line['self_bleu'], 
                line['self_bertscore'], 
                line['sbert'], 
                line['src_perplexity'], 
                line['pred_perplexity'], 
                line['src_syllable_per_word'], 
                line['pred_syllable_per_word'], 
                line['commonlit_src'], 
                line['commonlit_pred'],
                line['bleu'], 
                line['bertscore'], 
                line['sari']
            ]
        }
        # Make masks for available scores
        score_mask = [0 for i in range(self.n_supervision)]
        for i in range(len(out['scores'])):
            score_mask[i] = 1
        out['score_mask'] = score_mask

        # Zero-pad for SARI, referenced BLEU/BERTScore, Human Ratings
        while len(out['scores']) < self.n_supervision:
            out['scores'].append(0)
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
    
def make_pretraining_stage2_loader(path, tokenizer, batch_size, shuffle=True):
    dataset = PretrainingStage2Dataset(path, tokenizer)
    print(f'Making dataloader: {path}')
    print(f'# samples: {len(dataset)}')
    loader = DataLoader(dataset, batch_size=batch_size ,shuffle=shuffle, collate_fn=mr_collate)
    return loader