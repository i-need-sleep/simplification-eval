import os
import lzma
import copy

import torch
import nltk
import transformers
import pandas as pd

import utils.globals as uglobals

class OpenWebTextDataset(torch.utils.data.Dataset):
    def __init__(self, out_format, n_volumns=10, debug=False):
        self.out_format = out_format
        self.n_volumns= n_volumns
        self.debug = debug

        nltk.download('punkt')
        
        # Load documents into a long string
        self.text = self.load_volumns()

        # Tokenize into sentences
        self.sentences = self.build_sentencs()

        if self.out_format == 'prompt':
            self.data = self.build_prompts()
        elif self.out_format == 'save_txt':
            self.save_txt()
        else:
            raise NotImplementedError
        
    def __len__(self):
        return len(self.data) 

    def __getitem__(self, i):
        return self.data[i]
        
    def load_volumns(self):
        # Resolve the data dir
        if os.path.isdir(uglobals.OPENWEBTEXT_DIR):
            data_dir = uglobals.OPENWEBTEXT_DIR
        else:
            data_dir = uglobals.OPENWEBTEXT_DIR_ALT

        data_str = ''
        for file_idx, file_name in enumerate(os.listdir(data_dir)):
            if file_idx >= self.n_volumns:
                break
            with lzma.open(f'{data_dir}/{file_name}', 'r') as f:
                for line_idx, line in enumerate(f):
                    if line_idx == 0:
                        continue

                    if self.debug and line_idx > 5:
                        break

                    line_text = line.decode('UTF-8').replace('\x00', '').strip()
                    if line_text != '\n':
                        data_str += f'{line_text} '.replace('\n', '')
        
        print(f'Loaded {self.n_volumns} volumns from OpenWebText')
        return data_str
    
    def build_sentencs(self):
        sentences = nltk.tokenize.sent_tokenize(self.text)
        print(f'# Sentences {len(sentences)}')
        return sentences
    
    def save_txt(self):
        # Save the sentences line-by-line into a txt file
        out = ''
        for sent in self.sentences:
            out += sent + '\n'
        with open(f'{uglobals.SIM_OPENWEBTEXT_DIR}/original.en', 'w', encoding='utf-8') as f:
            f.write(out)
    
    def build_prompts(self, n_shot=5, start_idx=23, interval=100):
        # Load in-context examples from TurkCorpus
        src_path = f'{uglobals.TURKCORPUS_DIR}/tune.8turkers.tok.norm'
        ref_path = f'{uglobals.TURKCORPUS_DIR}/tune.8turkers.tok.turk.0'
        with open(src_path, 'r', encoding='utf-8') as f:
            src_text = f.readlines()
            src_sents = src_text[start_idx: start_idx + n_shot * interval: interval]
        with open(ref_path, 'r', encoding='utf-8') as f:
            ref_text = f.readlines()
            ref_sents = ref_text[start_idx: start_idx + n_shot * interval: interval]

        # Build 5-shot prompts for GPT3.5 turbo
        system_message = {
            "role": "system", 
            "content": 'You are a helpful assistant that simplifies English sentences, making them easier to read while preserving key meanings.'
        }
        user_prompt = 'Follow the examples and simplify the sentence, making it easier to read while preserving key meanings. Reply with only the simplified sentence. '

        for src_sent, ref_sent in zip(src_sents, ref_sents):
            user_prompt += f'Sentence: {src_sent} Simplification: {ref_sent} '
        prompt_base = [
            system_message,
            {"role": "user", "content": user_prompt}
        ]

        out = []
        for sent in self.sentences:
            prompt = copy.deepcopy(prompt_base)
            prompt[1]['content'] += f'Sentence: {sent} Simplification: '
            out.append({
                'prompt': prompt,
                'src': sent
            })
        return out
    
def convert_for_commonlit(path):
    df = pd.read_csv(path)
    srcs = df['src'].tolist()
    preds = df['pred'].tolist()
    indices = [i for i in range(len(srcs))]
    
    df_src = pd.DataFrame({
        'id': indices,
        'url_legal': ['' for _ in indices],
        'license': ['' for _ in indices],
        'excerpt': srcs,
    })
    df_src.to_csv(path.replace('csv', '_commonlit_src.csv'), index=False)

    df_pred = pd.DataFrame({
        'id': indices,
        'url_legal': ['' for _ in indices],
        'license': ['' for _ in indices],
        'excerpt': preds,
    })
    df_pred.to_csv(path.replace('csv', '_commonlit_pred.csv'))

def muss_to_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    src = []
    pred = []
    for idx, line in enumerate(lines):
        if idx % 3 == 1:
            src.append(line[9:].strip())
        elif idx % 3 == 2:
            pred.append(line[11:].strip())
    df = pd.DataFrame({
        'src': src,
        'pred': pred
    })
    df.to_csv(path.replace('.out', '.csv'))

def piece_commonlit_supervision(names):
    # Piece together commonlit scores (src and pred) with other supervision signals
    for name in names:
        data_path = f'{uglobals.PROCESSED_DIR}/openwebtext/supervision/{name}_supervision.csv'
        commonlit_src_path = f'{uglobals.PROCESSED_DIR}/openwebtext/supervision/{name}_src_commonlit_out.csv'
        commonlit_pred_path = f'{uglobals.PROCESSED_DIR}/openwebtext/supervision/{name}_pred_commonlit_out.csv'
        out_path = f'{uglobals.PROCESSED_DIR}/openwebtext/train/{name}.csv'

        out = {}
        df = pd.read_csv(data_path)
        for col in df.columns:
            if col != 'Unnamed: 0':
                out[col] = df[col].tolist()
        out['commonlit_src'] = pd.read_csv(commonlit_src_path)['target'].tolist()
        out['commonlit_pred'] = pd.read_csv(commonlit_pred_path)['target'].tolist()
        
        pd.DataFrame(out).to_csv(out_path)
        print(f'Saved at {out_path}')