import os

import pandas as pd

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