import os
import copy
import json

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

    for fold_idx in range(n_fold):
        # Select original sentences for the dev/test splits
        dev_indices = np.random.choice([i for i in range(60)], size = dev_size, replace = False).tolist()
        indices = []
        for i in range(60):
            if i not in dev_indices:
                indices.append(i)
        test_indices = np.random.choice(indices, size = test_size, replace = False).tolist()

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

def get_concordant_discordant_filtered(a, b, min_diff=5):

    con = 0
    dis = 0

    # The LENS paper uses only pairs where all three annotators agree with the ranking order
    # and the unnormalised score difference is larger than 5

    # If by that, they mean the score difference is larger than 5 for each annotaor:
    if False:
        for i in range(len(a)):
            for j in range(0, i):

                # Filter out invalid pairs
                filtered = False
                for annotator_idx in range(1, 4):
                    diff = b.iloc[j][f'rating_{annotator_idx}'] - b.iloc[i][f'rating_{annotator_idx}']

                    # Filter out cases where score diff <= 5
                    if abs(diff) <= min_diff:
                        filtered = True

                    # Make sure that all annotators agree with the order
                    if annotator_idx == 1:
                        larger = diff > 0
                    else:
                        if (diff > 0) != larger:
                            filtered = True
                            break
                if filtered:
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

    # ... If by that, they mean the average score difference is larger than 5:
    else:
        for i in range(len(a)):
            for j in range(0, i):

                # Filter
                filtered = False
                diffs = []
                for annotator_idx in range(1, 4):
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
                avg = sum(diffs) / len(diffs)
                if abs(avg) <= min_diff:
                    filtered = True
                    
                if filtered:
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

    print(f'Concordant: {con}, discordant: {dis}')
    return (con - dis) / (con + dis)

def test_simpeval_2022(score_function=None, score_path=''):
    df = pd.read_excel(f'{uglobals.STAGE3_DIR}/simpeval_2022.xlsx')
    df_original = copy.deepcopy(df)
    # Using Human 1 Writing as the reference and Human 2 Writing as the oracle output
    filtered_indices = df['system'] != 'Human 1 Writing'
    df = df[df['system'] != 'Human 1 Writing']

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
    print(f'Kendall Tau-like (filtered pairs): {kendall}')

    # Pearson Corrlation
    avg_annotator_score = (np.array(df['rating_1_zscore']) + np.array(df['rating_2_zscore']) + np.array(df['rating_3_zscore'])) / 3
    pearson = pearsonr(scores, avg_annotator_score).statistic
    print(f'Pearson correlation: {pearson}')

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

def get_fkgl(srcs, preds, refs):
    import textstat
    scores = []
    for idx, (src, pred, ref) in enumerate(zip(srcs, preds, refs)):
        score =  score = textstat.syllable_count(pred) / textstat.lexicon_count(pred)
        scores.append(score)
    return scores

def get_bleurt_pretrained(srcs, preds, refs, checkpoint='lucadiliello/BLEURT-20-D12'):
    from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
    bleurt = BleurtForSequenceClassification.from_pretrained(checkpoint) 
    device = torch.device('cpu')
    bleurt.to(device)
    bleurt.eval()
    tokenizer = BleurtTokenizer.from_pretrained(checkpoint)

    with torch.no_grad():
        inputs = tokenizer(preds, [ref[0] for ref in refs], padding='longest', return_tensors='pt').to(device)
        scores = bleurt(**inputs).logits.flatten().cpu().tolist()
    return scores