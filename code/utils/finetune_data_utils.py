import os
import copy
import json
import random
import re

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
import evaluate
import Levenshtein

import utils.globals as uglobals

def simpeval_asset_make_splits(dev_size=10):
    df = pd.read_csv(f'{uglobals.STAGE3_DIR}/simpeval_asset.csv')
    
    # Select original sentences for the dev set
    dev_indices = np.random.choice([i for i in range(100)], size = dev_size, replace = False).tolist()
    
    train = [[], [], [], []] # src, pred, score, ref
    dev = [[], [], [], []]
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

        # Resolve the reference
        original_id = df.iloc[i]['original_id']
        same_id = df['original_id'] == original_id
        human_generated = df['system'].isin(['asset.test.simp'])
        ref = df[human_generated & same_id]['generation'].tolist()[0]
        lis[3].append(ref)

    train_out = {
        'src': train[0],
        'pred': train[1],
        'score': train[2],
        'ref': train[3]
    }
    dev_out = {
        'src': dev[0],
        'pred': dev[1],
        'score': dev[2],
        'ref': dev[3]
    }
    pd.DataFrame(train_out).to_csv(f'{uglobals.STAGE3_PROCESSED_DIR}/simpeval_asset_train.csv', index=False)
    pd.DataFrame(dev_out).to_csv(f'{uglobals.STAGE3_PROCESSED_DIR}/simpeval_asset_dev.csv', index=False)

def process_simpeval_2022():
    df = pd.read_excel(f'{uglobals.STAGE3_DIR}/simpeval_2022.xlsx')
    
    test = [[], [], [], []]
    for i in range(len(df)):
        line = df.iloc[i]
        
        test[0].append(line['original'])
        test[1].append(line['generation'])

        # Resolve the reference
        original_id = df.iloc[i]['original_id']
        same_id = df['original_id'] == original_id
        human_generated = df['system'].isin(['Human 1 Writing'])
        ref = df[human_generated & same_id]['generation'].tolist()[0]
        test[2].append(ref)

        scores = [line[f'rating_{str(i+1)}_zscore'] for i in range(3)]
        test[3].append(sum(scores) / len(scores))

    out = {
        'src': test[0],
        'pred': test[1],
        'ref': test[2],
        'score': test[3],
    }
    pd.DataFrame(out).to_csv(f'{uglobals.STAGE3_PROCESSED_DIR}/simpeval_2022.csv', index=False)

def process_simp_da(dev_size=5, test_size=15, n_fold=4):
    df = pd.read_excel(f'{uglobals.STAGE3_DIR}/simpDA_2022.xlsx')
    
    # Normalize for each annotator
    for annotator_idx in range(11):
        for score_name in ['adequacy', 'fluency', 'simplicity']:
            scores = df[df['WorkerId'] == annotator_idx][f'Answer.{score_name}']
            if scores.std() == 0:
                std = 1 #Handle cases where all scores are the same
            else:
                std = scores.std()
            df.loc[df['WorkerId'] == annotator_idx, f'Answer.{score_name}'] = (scores - scores.mean()) / std

    # Use Human 1 Writing as the reference 
    # Both Human 1 Writing and Human 2 Writing are treated as oracle outputs
    df_copy = copy.deepcopy(df)
    df_refs = df_copy[df_copy['Input.system'] == 'Human 1 Writing']

    def save_splits(lis, name, fold_idx):
        for idx, score_name in enumerate(['adequacy', 'fluency', 'simplicity']):
            out = {
                'src': lis[0],
                'pred': lis[1],
                'ref': lis[2],
                'score': lis[3 + idx]
            }
            pd.DataFrame(out).to_csv(f'{uglobals.STAGE3_PROCESSED_DIR}/simp_da_fold{fold_idx}_{name}_{score_name}.csv')

    # Scramble the indices for the source sentences
    src_indices = [i for i in range(60)]
    random.shuffle(src_indices)

    for fold_idx in range(n_fold):
        # Select original sentences for the dev/test splits
        test_indices = src_indices[: test_size]
        dev_indices  = src_indices[test_size: test_size + dev_size]
        train_indices  = src_indices[test_size + dev_size : ]
        src_indices = src_indices[test_size: ] + src_indices[: test_size]

        train = [[] for _ in range(6)]
        dev = [[] for _ in range(6)]
        test = [[] for _ in range(6)]
        for i in range(int(len(df) / 3)):
            line = df.iloc[3 * i]
            lines = df.iloc[3 * i: 3 * i + 3]
            adequacy = np.average(np.array(lines.iloc[:, -3]))
            fluency = np.average(np.array(lines.iloc[:, -2]))
            simplicity = np.average(np.array(lines.iloc[:, -1]))

            # Retrieve the reference
            ref = df_refs[df_refs['Input.id'] == line['Input.id']]['Input.simplified'].tolist()[0]

            lis = train
            if line['Input.id'] in dev_indices:
                lis = dev
            elif line['Input.id'] in test_indices:
                lis = test
                # No reference for the test set
                if line['Input.system'] == 'Human 1 Writing':
                    continue

            lis[0].append(line['Input.original'])
            lis[1].append(line['Input.simplified'])
            lis[2].append(ref)
            lis[3].append(adequacy)
            lis[4].append(fluency)
            lis[5].append(simplicity)
    
        save_splits(train, 'train', fold_idx)
        save_splits(dev, 'dev', fold_idx)
        save_splits(test, 'test', fold_idx)

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

class FinetuneDatasetBLEURT(Dataset):
    def __init__(self, aggregated_path):
        self.df = pd.read_csv(aggregated_path)
        self.df.fillna('Unk', inplace=True)

    def __len__(self):
        return len(self.df) 

    def __getitem__(self, i):
        line = self.df.iloc[i]
        
        # Concat src and pred 
        out = {
            'src': line['src'],
            'pred': line['pred'],
            'ref': line['ref'],
            'scores': line['score']
            }
        return out

def mr_collate_bleurt(batch):
    for idx, line in enumerate(batch):
        if idx == 0:
            src = [line['pred']]
            pred = [line['src']]
            ref = [line['ref']]
            scores = torch.tensor(line['scores']).unsqueeze(0)
        else:
            src.append(line['src'])
            pred.append(line['pred'])
            ref.append(line['ref'])
            scores = torch.cat((scores, torch.tensor(line['scores']).unsqueeze(0)), dim=0).float()
    return src, pred, ref, scores
    
def make_finetuning_loader_bleurt(path, batch_size, shuffle=True):
    dataset = FinetuneDatasetBLEURT(path)
    print(f'Making BLEURT dataloader: {path}')
    print(f'# samples: {len(dataset)}')
    loader = DataLoader(dataset, batch_size=batch_size ,shuffle=shuffle, collate_fn=mr_collate_bleurt)
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

def get_concordant_discordant(a, b, n_sys=5):
    con = 0
    dis = 0
    for i in range(len(a)):
        for j in range(i, len(a)):
            # Consider only system outputs from the same source sentence
            if i // n_sys != j // n_sys:
                continue

            # Consider only the same type of simplification
            
            if (a[j] - a[i]) * (b[j] - b[i]) > 0:
                con += 1
            else:
                dis += 1
    return (con - dis) / (con + dis)

def get_concordant_discordant_type(scores, df):
    con = 0
    dis = 0

    original_df = pd.read_csv(f'{uglobals.STAGE3_DIR}/simpeval_2022.csv')

    for i in range(len(df)):
        for j in range(i):

            # Consider only system outputs from the same source sentence
            if df.iloc[i]['src'] != df.iloc[j]['src']:
                continue            
            
            def find_max_overlap(str, df):
                # Given a string, return the row in df that has the highest levenshtein distance
                min_dist = 100
                for i in range(len(df)):
                    dist = Levenshtein.distance(str, df.iloc[i]['generation'])
                    if dist < min_dist:
                        min_dist = dist
                        idx = i
                # if min_dist > 3:
                #     print(min_dist)
                return idx

            # Resolve the original annotations
            type_i = original_df.iloc[find_max_overlap(df.iloc[i]['pred'], original_df)]['sentence_type']
            type_j = original_df.iloc[find_max_overlap(df.iloc[j]['pred'], original_df)]['sentence_type']

            # Consider only the same type of simplification
            if type_i != type_j:
                continue

            if (scores[j] - scores[i]) * (df.iloc[j]['score'] - df.iloc[i]['score']) > 0:
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
    from lens.lens_score import LENS

    try:
        df = pd.read_csv(path)
    except:
        df = pd.read_excel(path)
    
    metric = LENS(uglobals.LENS_DIR, rescale=True)

    if 'score' in df.columns.tolist():
        with torch.no_grad():
            lens_scores = metric.score(df['src'].tolist(), df['pred'].tolist(), [[ref] for ref in df['ref'].tolist()], batch_size=8, gpus=0)

        # Pearson Corrlation
        pearson = pearsonr(lens_scores, df['score']).statistic
        print(f'Pearson correlation: {pearson}')

        # kendall tau-like
        kendall = get_concordant_discordant(lens_scores, df['score'])
        print(f'Kendall Tau-like: {kendall}')
    else:
        with torch.no_grad():
            lens_scores = metric.score(df['original'].tolist(), df['generation'].tolist(), [[ref] for ref in df['ref'].tolist()], batch_size=8, gpus=0)

        # Kendall tau-like with pairs where all annotators agree with the order and unormalized score differences > 5
        kendall = get_concordant_discordant_filtered(lens_scores, df)
        print(f'Kendall Tau-like (filtered pairs): {kendall}')

def test_sari(path):
    from easse.sari import corpus_sari

    try:
        df = pd.read_csv(path)
    except:
        df = pd.read_excel(path)
    
    if 'score' in df.columns.tolist():
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
    else:
        scores = []
        for i in range(len(df)):
            src = df.iloc[i]['original']
            pred = df.iloc[i]['generation']
            ref = df.iloc[i]['ref']
            score = corpus_sari(orig_sents=[src], sys_sents=[pred], refs_sents=[[ref]])
            scores.append(score)

        # Kendall tau-like with pairs where all annotators agree with the order and unormalized score differences > 5
        kendall = get_concordant_discordant_filtered(scores, df)
        print(f'Kendall Tau-like (filtered pairs): {kendall}')

# def test_bleu(path):

#     bleu = evaluate.load('bleu')
    
    
#     try:
#         df = pd.read_csv(path)
#     except:
#         df = pd.read_excel(path)
        
    
#     if 'score' in df.columns.tolist():
#         scores = []
#         for i in range(len(df)):
#             src = df.iloc[i]['src']
#             pred = df.iloc[i]['pred']
#             ref = df.iloc[i]['ref']
#             score = bleu.compute(predictions = [pred], references = [ref])['bleu']
#             scores.append(score)

#         # Pearson Corrlation
#         pearson = pearsonr(scores, df['score']).statistic
#         print(f'Pearson correlation: {pearson}')

#         # kendall tau-like
#         kendall = get_concordant_discordant(scores, df['score'])
#         print(f'Kendall Tau-like: {kendall}')
#     else:
#         scores = []
#         for i in range(len(df)):
#             src = df.iloc[i]['original']
#             pred = df.iloc[i]['generation']
#             ref = df.iloc[i]['ref']
#             score = bleu.compute(predictions = [pred], references = [ref])['bleu']
#             scores.append(score)

#         # Kendall tau-like with pairs where all annotators agree with the order and unormalized score differences > 5
#         kendall = get_concordant_discordant_filtered(scores, df)
#         print(f'Kendall Tau-like (filtered pairs): {kendall}')

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

def simp_da_normalize_and_average():
    df = pd.read_excel(f'{uglobals.STAGE3_DIR}/simpDA_2022.xlsx')

    # Normalize for 
    max_worker_idx = max(df['WorkerId'].tolist())

    for workder_idx in range(max_worker_idx):
        for measure in ['adequacy', 'fluency', 'simplicity']:
            mean = np.mean(df[df["WorkerId"] == workder_idx][f'Answer.{measure}'])
            std = np.std(df[df["WorkerId"] == workder_idx][f'Answer.{measure}'])
            print(mean, std)
            df.loc[df["WorkerId"] == workder_idx, f'Answer.{measure}'] = (df[df["WorkerId"] == workder_idx][f'Answer.{measure}'] - mean) / std
    print(df)

def test_bleu_simpda():
    from easse.sari import corpus_sari

    bleu = evaluate.load('bleu')
    
    df = pd.read_csv(f'{uglobals.STAGE3_DIR}/simplicity_DA.csv')

    # Load the reference simplifications
    refs = [] # [[sent0_ref0, ...], ...]
    for i in range(10):
        with open(f'{uglobals.STAGE3_DIR}/asset/asset.test.simp.{i}', encoding='utf-8') as f:
            data = f.readlines()
            refs.append(data)
    
    scores = []
    for i in range(len(df)):
        src = df.iloc[i]['orig_sent']
        pred = df.iloc[i]['simp_sent']
        sent_id = df.iloc[i]['sent_id']
        # ref = [r[sent_id - 1] for r in refs]
        ref = [[r[sent_id - 1]] for r in refs]
        # score = bleu.compute(predictions = [pred], references = [ref])['bleu']
        score = corpus_sari(orig_sents=[src], sys_sents=[pred], refs_sents=ref)
        scores.append(score)

    # Pearson Corrlation
    for measure in ['fluency', 'meaning', 'simplicity']:
        print(measure)
        pearson = pearsonr(scores, df[f'{measure}_zscore']).statistic
        print(f'Pearson correlation: {pearson}')
        pearson = pearsonr(scores, df[measure]).statistic
        print(f'Pearson correlation: {pearson}')

def test_bleu_simpeval_2022():
    # Evaluate BLEU's correlation with human scores
    # This should correspond to the tao_all value on Table 2 of the LENS paper

    # The BLEU implemntation from Huggingface's evaluate package
    # bleu = evaluate.load('bleu')
    from torchmetrics import BLEUScore
    bleu = BLEUScore()
    
    # SimpEval 2022 as provided in the LENS repo
    df = pd.read_excel(f'{uglobals.STAGE3_DIR}/simpeval_2022.xlsx')
    df_original = copy.deepcopy(df)

    # The dataset contains two human simplifications for each source sentence.
    # They use one as the reference and the other as the oracle output.
    # I use Human 2 Writing as the reference and Human 1 Writing as the oracle output here. I've also tried the other way around.
    df = df[df['system'] != 'Human 2 Writing']
    
    scores = []
    for i in range(len(df)):
        pred = df.iloc[i]['generation']

        # Resolve the reference
        original_id = df.iloc[i]['original_id']
        human_generated = df_original['system'].isin(['Human 2 Writing'])
        same_id = df_original['original_id'] == original_id
        refs = df_original[human_generated & same_id]['generation'].tolist()
        
        # Compute BLEU under the default settings
        # score = bleu.compute(predictions = [pred], references = [refs])['bleu']
        score = bleu([pred], [refs]).item()
        scores.append(score)

    # Kendall tau-like with pairs where all annotators agree with the ranking order and unormalized score differences > 5
    kendall = get_concordant_discordant_filtered(scores, df)
    print(f'Kendall Tau-like (filtered pairs): {kendall}')

def get_concordant_discordant_filtered(a, b, measure, min_diff=5):

    con = 0
    dis = 0
    n_filtered = 0

    original_df = pd.read_csv(f'{uglobals.STAGE3_DIR}/simpDA_2022.csv')

    # The LENS paper uses only pairs where all three annotators agree with the ranking order
    # and the unnormalised score difference is larger than 5

    # The score difference should be larger than 5 for at least two of the three annotators. 
    for i in range(len(a)):
        for j in range(0, i):

            # Consider only system outputs for the same source sentence
            if b.iloc[i]['src'] != b.iloc[j]['src']:
                continue

            def swap_non_alpha(s):
                return re.sub(r'\W+', '', s)

            # Resolve the original annotations
            annotations_i = original_df[swap_non_alpha(original_df['Input.simplified']) == swap_non_alpha(b.iloc[i]['pred'])]
            annotations_j = original_df[swap_non_alpha(original_df['Input.simplified']) == swap_non_alpha(b.iloc[j]['pred'])]
            
            print(annotations_i)
            print(annotations_j)
            exit()

            # Filter
            filtered = False
            diffs = []
            for annotator_idx in range(3):
                diff = b.iloc[j][f'rating_{annotator_idx}'] - b.iloc[i][f'rating_{annotator_idx}']
                diffs.append(diff)

                # Make sure that all annotators agree with the order
                if annotator_idx == 1:
                    larger = diff > 0
                else:
                    if (diff > 0) != larger:
                        filtered = True
                        break

            # Make sure the average score diff is larger than 5
            larger_than_5_ctr = 0
            for diff in diffs:
                if abs(diff) > min_diff:
                    larger_than_5_ctr += 1
            if larger_than_5_ctr < 2:
                filtered = True
                
            if filtered:
                n_filtered += 1
                continue

            if larger:
                larger = 1
            else:
                larger = -1

            # Count concordanct and discordant pairs
            if (a[j] - a[i]) * larger > 0:
                con += 1
            else:
                dis += 1

    print(f'Concordant: {con}, discordant: {dis}, filtered: {n_filtered}')
    return (con - dis) / (con + dis)

def test_simpeval_2022(score_function=None, score_path='', sent_type='Splittings'):
    df = pd.read_excel(f'{uglobals.STAGE3_DIR}/simpeval_2022.xlsx')
    df_original = copy.deepcopy(df)
    # Using Human 1 Writing as the reference and Human 2 Writing as the oracle output
    filtered_indices = df['sentence_type'] == sent_type
    df = df[df['sentence_type'] == sent_type]

    srcs, preds, refs = [], [], []
    for i in range(len(df)):
        src = df.iloc[i]['original']
        pred = df.iloc[i]['generation']

        # Resolve the reference
        original_id = df.iloc[i]['original_id']
        human_generated = df_original['system'].isin(['Human 1 Writing'])
        same_id = df_original['original_id'] == original_id
        ref = df_original[human_generated & same_id]['generation'].tolist()

        srcs.append(src)
        preds.append(pred)
        refs.append(ref)
    
    if score_path == '':
        scores = score_function(srcs, preds, refs)
    else:
        with open(score_path, 'r') as f:
            scores_in = json.load(f)

        # Filter out the scores of the reference
        scores = []
        for i in range(len(scores_in)):
            if filtered_indices[i]:
                scores.append(scores_in[i])

    # Kendall tau-like with pairs where all annotators agree with the ranking order and unormalized score differences > 5
    kendall = get_concordant_discordant_filtered(scores, df)
    print(f'Kendall Tau-like (filtered pairs): {"%.3f" % kendall}')

    # Pearson Corrlation
    avg_annotator_score = (np.array(df['rating_1_zscore']) + np.array(df['rating_2_zscore']) + np.array(df['rating_3_zscore'])) / 3
    pearson = pearsonr(scores, avg_annotator_score).statistic
    print(f'Pearson correlation: {"%.3f" % pearson}')

    # Kendall tau-like on all the data
    kendall = get_concordant_discordant(scores, avg_annotator_score)
    print(f'Kendall Tau-like (all pairs): {"%.3f" % kendall}')

    return

def get_bleu_scores(srcs, preds, refs):
    from torchmetrics import BLEUScore
    bleu = BLEUScore()

    scores = []
    for idx, (pred, ref) in enumerate(zip(preds, refs)):
        score = bleu([pred], [ref]).item()
        scores.append(score)
    return scores

def get_lens_scores(srcs, preds, refs):
    from lens.lens_score import LENS
    metric = LENS(uglobals.LENS_DIR, rescale=True)
    scores = metric.score(srcs, preds, refs, batch_size=8, gpus=0)
    return scores

def get_sari_scores(srcs, preds, refs):
    from easse.sari import corpus_sari
    scores = []
    for idx, (src, pred, ref) in enumerate(zip(srcs, preds, refs)):
        score = corpus_sari(orig_sents=[src], sys_sents=[pred], refs_sents=[ref])
        scores.append(score)
    return scores

def get_bertscores_f1(srcs, preds, refs):
    bertscore = evaluate.load('bertscore')
    scores = []
    for idx, (src, pred, ref) in enumerate(zip(srcs, preds, refs)):
        score = bertscore.compute(predictions = [pred], references = [ref], lang='en')['f1'][0]
        scores.append(score)
    return scores

def get_bertscores_precision(srcs, preds, refs):
    bertscore = evaluate.load('bertscore')
    scores = []
    for idx, (src, pred, ref) in enumerate(zip(srcs, preds, refs)):
        score = bertscore.compute(predictions = [pred], references = [ref], lang='en')['precision'][0]
        scores.append(score)
    return scores

def get_self_bertscore_f1(srcs, preds, refs):
    bertscore = evaluate.load('bertscore')
    scores = []
    for idx, (src, pred, ref) in enumerate(zip(srcs, preds, refs)):
        score = bertscore.compute(predictions = [pred], references = [src], lang='en')['f1'][0]
        scores.append(score)
    return scores

def get_self_bertscore_precision(srcs, preds, refs, bertscore=None):
    if bertscore == None:
        bertscore = evaluate.load('bertscore')
    scores = []
    for idx, (src, pred, ref) in enumerate(zip(srcs, preds, refs)):
        score = bertscore.compute(predictions = [pred], references = [src], lang='en')['precision'][0]
        scores.append(score)
    return scores

def get_fkgl(srcs, preds, refs):
    import textstat
    scores = []
    for idx, (src, pred, ref) in enumerate(zip(srcs, preds, refs)):
        # score = textstat.syllable_count(pred) / textstat.lexicon_count(pred)
        score = textstat.flesch_kincaid_grade(pred)
        scores.append(score)
    return scores

def get_sle(srcs, preds, refs):

    from sle.scorer import SLEScorer
    scorer = SLEScorer("liamcripwell/sle-base")

    scores = []
    for idx, (src, pred, ref) in enumerate(zip(srcs, preds, refs)):
        score = scorer.score([pred], inputs=[src])['sle_delta'][0]
        scores.append(score)
    return scores

def get_bets(srcs, preds, refs):
    import transformers
    from models.bets.metric import p_simp_score, r_meaning_score

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = f'../../bets/checkpoints/comparative0.ckpt'
    ensemble_weights = []

    model = 'roberta-large'

    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    lm_model =  transformers.AutoModel.from_pretrained(model)

    ranker = torch.load(checkpoint_path, map_location=device)
    ranker.eval()
    ranker.to(device)

    if ensemble_weights == []:
        simplicity_weight = 0.508
        meaning_weight = 2.944
    else:
        simplicity_weight = ensemble_weights[0]
        meaning_weight = ensemble_weights[1]

    scores = []
    for idx, (src, pred, ref) in enumerate(zip(srcs, preds, refs)):
        try:
            simplicity_score = p_simp_score(src, pred, tokenizer, lm_model, ranker, device)
        except:
            print('simp error')
            simplicity_score = 0

        try:
            meaning_score = r_meaning_score(src, pred, tokenizer, lm_model, device)
        except:
            print('meaning wrong')
            meaning_score = 0

        score = simplicity_weight * simplicity_score + meaning_weight * meaning_score
        scores.append(score)
    return scores

def get_bleurt_pretrained(srcs, preds, refs, checkpoint='lucadiliello/BLEURT-20-D12', bleurt=None, tokenizer=None):
    if bleurt == None:
        from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
        bleurt = BleurtForSequenceClassification.from_pretrained(checkpoint) 
        device = torch.device('cpu')
        bleurt.to(device)
        bleurt.eval()
        tokenizer = BleurtTokenizer.from_pretrained(checkpoint)
    
    device = torch.device('cpu')

    with torch.no_grad():
        inputs = tokenizer(preds, [ref[0] for ref in refs], padding='longest', return_tensors='pt').to(device)
        scores = bleurt(**inputs).logits.flatten().cpu().tolist()
    return scores

def get_bleurt_finetuned_simpeval(srcs, preds, refs, checkpoint='../results/checkpoints/simpeval/bleurt.bin'):
    from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
    original_checkpoint = 'lucadiliello/BLEURT-20-D12'

    bleurt = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20-D12') 
    device = torch.device('cpu')
    print(f'Loading BLEURT checkpoint: {checkpoint}')
    bleurt.load_state_dict(torch.load(checkpoint, map_location=device)['model_state_dict'])
    bleurt.to(device)
    bleurt.eval()
    tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20-D12')


    with torch.no_grad():
        inputs = tokenizer(preds, [ref[0] for ref in refs], padding='longest', return_tensors='pt').to(device)
        scores = bleurt(**inputs).logits.flatten().cpu().tolist()
    return scores

def test_da(score_function, data_root=f'{uglobals.STAGE3_PROCESSED_DIR}/humanlikert_splits/simp_da', n_fold=4, multi_ref=True, filter_kendall=False, save_name='', set_measure='', set_fold=''):
    out_str = ''
    dfs = []
    for measure in ['adequacy', 'fluency', 'simplicity']:
        if set_measure != '' and set_measure != measure:
            continue
        pearsons = []
        kendall_likes = []
        for fold_idx in range(n_fold):
            if set_fold != '' and set_fold != fold_idx:
                continue

            # Get loaders
            df = pd.read_csv(f'{data_root}_fold{fold_idx}_test_{measure}.csv')

            src = df['src'].tolist()
            pred = df['pred'].tolist()
            ref = df['ref'].tolist()
            ref = [[r] for r in ref]
            human_scores = df['score'].tolist()

            from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
            bleurt = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20-D12') 
            device = torch.device('cpu')
            bleurt.to(device)
            bleurt.eval()
            tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20-D12')
            
            if multi_ref:
                scores = []
                ref = [r[0][2: -2].split("', '") for r in ref]
                for s, p, r in zip(src, pred, ref):
                    line_scores = []
                    for sub_r in r:
                        line_scores.append(score_function([s], [p], [[sub_r]], bleurt=bleurt, tokenizer=tokenizer)[0])
                    scores.append(np.mean(line_scores))
            else:
                scores = score_function(src, pred, ref)

            if save_name != '':
                df_dict = {
                    'src': src,
                    'pred': pred,
                    'ref': [r[0] for r in ref],
                    'human_scores': human_scores,
                    'score': scores
                }
                df = pd.DataFrame(df_dict)
                dfs.append(df)

            # Pearson Corrlation
            pearson = pearsonr(scores, human_scores).statistic
            pearsons.append(pearson)

            # Kendall tau-like 
            if filter_kendall:
                kendall = get_concordant_discordant_type(scores, df)
            else:
                kendall = get_concordant_discordant(scores, human_scores)
            kendall_likes.append(kendall)
        
        if save_name != '':
            df = pd.concat(dfs)
            df.to_csv(f'{uglobals.OUTPUTS_DIR}/filtered/{save_name}.csv')
            return df

        print(measure)
        avg_pearson = '%.3f' % round(sum(pearsons) / len(pearsons), 3)
        avg_kendall = '%.3f' % round(sum(kendall_likes) / len(kendall_likes), 3)
        print('Pearson:', avg_pearson)
        print('Kendall-Tau-likes:', avg_kendall)
        std_pearsons = '%.3f' % np.std(np.array(pearsons))
        std_kendall = '%.3f' % np.std(np.array(kendall_likes))
        out_str += f'${avg_pearson} \pm {std_pearsons}$ & '# & ${avg_kendall} \pm {std_kendall}$ & '

    print(out_str[:-2] + '\\\\')
    return 

def simplicity_da_resolve_reference():
    ref_path = f'{uglobals.STAGE2_OUTPUTS_DIR}/systems/dress/WikiLarge/test/Reference'
    df_path = f'{uglobals.STAGE3_DIR}/simplicity_DA.csv'

    df = pd.read_csv(df_path)
    with open(ref_path) as f:
        refs = f.readlines()

    refs_out = []

    for i in range(len(df)):
        sent_id = df['sent_id'][i]
        refs_out.append(refs[sent_id - 1])
    
    df['ref'] = refs_out
    df.to_csv(df_path.replace('.csv', 'processed.csv'), index=False)

def process_simplicity_da(in_file, dev_size=60, test_size=60, n_fold=5):
    df = pd.read_csv(f'{uglobals.STAGE3_DIR}/{in_file}.csv')
    
    def save_splits(lis, name, fold_idx):
        for idx, score_name in enumerate(['adequacy', 'fluency', 'simplicity']):
            out = {
                'src': lis[0],
                'pred': lis[1],
                'ref': lis[2],
                'score': lis[3 + idx]
            }
            pd.DataFrame(out).to_csv(f'{uglobals.STAGE3_PROCESSED_DIR}/simp_da_fold{fold_idx}_{name}_{score_name}.csv')

    # Scramble the indices for the source sentences
    src_indices = list(set(df['sent_id'].tolist()))
    random.shuffle(src_indices)

    for fold_idx in range(n_fold):
        # Select original sentences for the dev/test splits
        test_indices = src_indices[: test_size]
        dev_indices  = src_indices[test_size: test_size + dev_size]
        train_indices  = src_indices[test_size + dev_size : ]
        src_indices = src_indices[test_size: ] + src_indices[: test_size]

        train = [[] for _ in range(6)]
        dev = [[] for _ in range(6)]
        test = [[] for _ in range(6)]
        for i in range(len(df)):
            line = df.iloc[i]

            lis = train
            if line['sent_id'] in dev_indices:
                lis = dev
            elif line['sent_id'] in test_indices:
                lis = test

            lis[0].append(line['orig_sent'])
            lis[1].append(line['simp_sent'])
            lis[2].append(line['ref'])
            lis[3].append(line['meaning_zscore'])
            lis[4].append(line['fluency_zscore'])
            lis[5].append(line['simplicity_zscore'])
    
        save_splits(train, 'train', fold_idx)
        save_splits(dev, 'dev', fold_idx)
        save_splits(test, 'test', fold_idx)

def get_referee(srcs, preds, refs):
    checkpoint = ''
    checkpoint = f'../../../simplification-eval/results/checkpoints/simpeval/{checkpoint}.bin'

    device = torch.device('cpu')
    model = DebertaForEval(uglobals.RORBERTA_MODEL_DIR, uglobals.RORBERTA_TOKENIZER_DIR, device, head_type='linear', backbone='deberta')

    model.load_state_dict(torch.load(checkpoint, map_location=device)['model_state_dict'])

    scores = []

    for src, pred in zip(srcs, preds):
        sent = [src + ' ' + model.tokenizer.sep_token + ' ' + pred]
        pred = model(sent)

        pred = pred[:, -1].reshape(-1).tolist()[0]
        scores.append(pred)
    return scores

def preprocess_human_likert():
    df = pd.read_csv(f'{uglobals.STAGE3_DIR}/human_likert.csv')

    for aspect in ['meaning', 'fluency', 'simplicity']:
        aspect_df = df[df['aspect'] == aspect]
        
        # Normalize the ratings for each worker_id
        for worker_id in set(aspect_df['worker_id'].tolist()):
            ratings = aspect_df[aspect_df['worker_id'] == worker_id]['rating'].tolist()
            mean = np.mean(ratings)
            std = np.std(ratings)
            df.loc[df['worker_id'] == worker_id, 'rating'] = (df[df['worker_id'] == worker_id]['rating'] - mean) / std
    
    df = df[df['simplification_type'] == 'human']
    df.drop(['worker_id', 'simplification_type'], axis=1, inplace=True)
    
    # For each sentence, for each aspect, take the average of all ratings
    sent_ids = df['sentence_id'].tolist()
    sent_ids = list(set(sent_ids))
    sent_ids.sort()

    out = {
        'sent_id': [],
        'orig_sent': [],
        'simp_sent': [],
        'ref': [],
        'fluency_zscore': [],
        'meaning_zscore': [],
        'simplicity_zscore': []
    }

    for sent_id in sent_ids:
        sent_id_df = df[df['sentence_id'] == sent_id]
        preds = sent_id_df['simplification'].tolist()
        preds = list(set(preds))
        
        for pred in preds:
            sent_df = sent_id_df[sent_id_df['simplification'] == pred]

            out['sent_id'].append(sent_id)

            out['orig_sent'].append(sent_df['source'].tolist()[0])
            out['simp_sent'].append(pred)
            out['ref'].append(sent_df['references'].tolist()[0])

            out['fluency_zscore'].append(np.mean(sent_df['rating'][sent_df['aspect'] == 'fluency'].tolist()))
            out['meaning_zscore'].append(np.mean(sent_df['rating'][sent_df['aspect'] == 'meaning'].tolist()))
            out['simplicity_zscore'].append(np.mean(sent_df['rating'][sent_df['aspect'] == 'simplicity'].tolist()))

    df_out = pd.DataFrame(out)
    df_out.to_csv(f'{uglobals.STAGE3_DIR}/human_likert_processed.csv', index=False)
    return