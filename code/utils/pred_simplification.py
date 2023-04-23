import os
import time

import openai
import pandas as pd

import utils.keychain as keychain
import utils.globals as uglobals

openai.api_key = keychain.OPENAI_API_KEY

def gptturbo_inference(dataset, save_interval=500):
    if not os.path.isdir(uglobals.SIM_OPENWEBTEXT_DIR):
        os.mkdir(uglobals.PROCESSED_DIR)
        os.mkdir(uglobals.SIM_OPENWEBTEXT_DIR)
    out_path = f'{uglobals.SIM_OPENWEBTEXT_DIR}/gpt_turbo.csv'

    out = {
        'src': [],
        'pred': []
    }

    for line_idx, line in enumerate(dataset):

        src = line['src']
        prompt = line['prompt']

        while True:
            try:
                c = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=prompt,
                    max_tokens=1024,
                    temperature=0,
                )
                break
            except:
                print("pausing")
                time.sleep(1)
                continue


        pred = c['choices'][0]['message']['content']


        out['src'].append(src)
        out['pred'].append(pred)

        if line_idx % save_interval == 0:
            df = pd.DataFrame(out)
            df.to_csv(out_path)

def gpt3_inference(engine, dataset, save_interval=100):
    if not os.path.isdir(uglobals.SIM_OPENWEBTEXT_DIR):
        os.mkdir(uglobals.PROCESSED_DIR)
        os.mkdir(uglobals.SIM_OPENWEBTEXT_DIR)
    out_path = f'{uglobals.SIM_OPENWEBTEXT_DIR}/gpt_{engine}.csv'

    out = {
        'src': [],
        'pred': []
    }

    for line_idx, line in enumerate(dataset):

        src = line['src']
        prompt = line['prompt'][1]['content']

        while True:
            try:
                c = openai.Completion.create(
                    model=engine,
                    prompt=prompt,
                    max_tokens=256,
                    temperature=0.8,
                )
                break
            except:
                print("pausing")
                time.sleep(1)
                continue

        pred = c['choices'][0]['text'].split('\n')[0]


        out['src'].append(src)
        out['pred'].append(pred)

        if line_idx % save_interval == 0:
            df = pd.DataFrame(out)
            df.to_csv(out_path)