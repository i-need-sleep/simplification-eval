import pandas as pd
import numpy as np
import evaluate
from sentence_transformers import SentenceTransformer
import textstat
import tqdm

import utils.globals as uglobals

class RefFreeSupervision():
    def __init__(self):
        self.bleu = evaluate.load('bleu')
        self.bertscore = evaluate.load('bertscore')
        self.sbert = SentenceTransformer(('sentence-transformers/all-distilroberta-v1'), cache_folder=uglobals.PRETRAINED_DIR)
        self.perplexity = evaluate.load("./utils/perplexity.py",  module_type= "measurement")
    
    def make_ref_free_supervisions(self, paths):
        for path in paths:
            self.make_ref_free_supervision(path)

    def make_ref_free_supervision(self, path):
        print(path)
        df = pd.read_csv(path)
        srcs = df['src'].tolist()
        preds = df['pred'].tolist()

        # Get supervision signals
        # Content preservation
        self_bleu = self.make_self_bleu(srcs, preds)
        self_bertscore = self.make_self_bertscore(srcs, preds)
        sbert = self.make_sbert(srcs, preds)

        # Fluency
        src_perplexity, pred_perplexity = self.make_perplexity(srcs, preds)

        # Readability
        src_syllable_per_word, pred_syllable_per_word = self.make_syllable_over_words(srcs, preds)

        write_path = path.replace('.csv', '_supervision.csv')
        df = pd.DataFrame({
            'src': srcs,
            'pred': preds,
            'self_bleu': self_bleu,
            'self_bertscore': self_bertscore,
            'sbert': sbert,
            'src_perplexity': src_perplexity,
            'pred_perplexity': pred_perplexity,
            'src_syllable_per_word': src_syllable_per_word,
            'pred_syllable_per_word': pred_syllable_per_word,
        })
        df.to_csv(write_path)
        print(f'Saved at {write_path}')
        return
    
    def make_self_bleu(self, srcs, preds):
        print('Calculating Self-BLEU')
        self_bleu = []
        for idx, (src, pred) in enumerate(tqdm.tqdm(zip(srcs, preds))):
            try:
                score = self.bleu.compute(predictions = [pred], references = [[src]])['bleu']
            except:
                score = 0
            self_bleu.append(score)
        return self_bleu
    
    def make_self_bertscore(self, srcs, preds):
        print('Calculating Self-BERTScore')
        self_bertscore = []
        for idx, (src, pred) in enumerate(tqdm.tqdm(zip(srcs, preds))):
            try:
                score = self.bertscore.compute(predictions = [pred], references = [src], lang='en')['f1'][0]
            except:
                score = 0.9
            self_bertscore.append(score)
        return self_bertscore
    
    def make_sbert(self, srcs, preds):
        print('Calculating SBERT cosine distance')
        sbert = []
        for idx, (src, pred) in enumerate(tqdm.tqdm(zip(srcs, preds))):
            try:
                score = self.get_cos_dist(pred, src)
            except:
                score = 0.9
            sbert.append(score)
        return sbert
    
    def get_cos_dist(self, s1, s2):
        embs = self.sbert.encode([s1, s2])
        out = np.sum(embs[0] * embs[1])
        return out
    
    def make_perplexity(self, srcs, preds):
        print('Calculating GPT-2 perplexity')
        src_perplexity, pred_perplexity = [], []
        for idx, (src, pred) in enumerate(tqdm.tqdm(zip(srcs, preds))):
            try:
                src_score = self.perplexity.compute(data=[src], model_id='gpt2')['perplexities'][0]
            except:
                src_score = 0.9
            try:
                pred_score = self.perplexity.compute(data=[pred], model_id='gpt2')['perplexities'][0]
            except:
                pred_score = 0.9
            src_perplexity.append(src_score)
            pred_perplexity.append(pred_score)
        return src_perplexity, pred_perplexity
    
    def make_syllable_over_words(self, srcs, preds):
        print('Calculating #syllables per word')
        src_scores, pred_scores = [], []
        for idx, (src, pred) in enumerate(tqdm.tqdm(zip(srcs, preds))):
            try:
                src_score = textstat.syllable_count(src) / textstat.lexicon_count(src)
            except:
                src_score = 0
            try:
                pred_score = textstat.syllable_count(pred) / textstat.lexicon_count(pred)
            except:
                pred_score = 0
            src_scores.append(src_score)
            pred_scores.append(pred_score)
        return src_scores, pred_scores

if __name__ == '__main__':
    paths = [f'{uglobals.PROCESSED_DIR}/openwebtext/muss.csv', f'{uglobals.PROCESSED_DIR}/openwebtext/gpt_curie.csv', f'{uglobals.PROCESSED_DIR}/openwebtext/augmented.csv']
    supervisor = RefFreeSupervision()
    supervisor.make_ref_free_supervisions(paths)
    