import pandas as pd
import numpy as np
import evaluate
from sentence_transformers import SentenceTransformer
import textstat
import tqdm

from easse.sari import corpus_sari

import utils.globals as uglobals

class RefBasedSupervision():
    def __init__(self):
        self.bleu = evaluate.load('bleu')
        self.bertscore = evaluate.load('bertscore')

    def make_ref_based_supervision(self, path):
        print(path)
        df = pd.read_csv(path)
        srcs = df['src'].tolist()
        preds = df['pred'].tolist()
        refs = df['ref'].tolist()

        # Get supervision signals
        bleu = self.make_bleu(preds, refs)
        bertscore = self.make_bertscore(preds, refs)
        sari = self.make_sari(srcs, preds, refs)

        write_path = path.replace('.csv', '_ref_based.csv')
        print(bleu, bertscore, sari)
        df = pd.DataFrame({
            'src': srcs,
            'pred': preds,
            'ref': refs,
            'bleu': bleu,
            'bertscore': bertscore,
            'sari': sari,
        })
        df.to_csv(write_path)
        print(f'Saved at {write_path}')
        return
    
    def make_bleu(self, preds, refs):
        print('Calculating BLEU')
        bleu = []
        for idx, (pred, ref) in enumerate(tqdm.tqdm(zip(preds, refs))):
            try:
                score = self.bleu.compute(predictions = [pred], references = [[ref]])['bleu']
            except:
                score = 0
            bleu.append(score)
        return bleu
    
    def make_bertscore(self, preds, refs):
        print('Calculating BERTScore')
        bertscore = []
        for idx, (pred, ref) in enumerate(tqdm.tqdm(zip(preds, refs))):
            try:
                score = self.bertscore.compute(predictions = [pred], references = [ref], lang='en')['f1'][0]
            except:
                score = 0.9
            bertscore.append(score)
        return bertscore
    
    def make_sari(self, srcs, preds, refs):
        print('Calculating SARI')
        sari = []
        for idx, (src, pred, ref) in enumerate(tqdm.tqdm(zip(srcs, preds, refs))):
            try:
                score = corpus_sari(orig_sents=[src], sys_sents=[pred], refs_sents=[[ref]])
            except:
                score = 0
            sari.append(score)
        return sari

if __name__ == '__main__':
    path = f'{uglobals.STAGE2_OUTPUTS_DIR}/aggregated_augmented.csv'
    supervisor = RefBasedSupervision()
    supervisor.make_ref_based_supervision(path)
    