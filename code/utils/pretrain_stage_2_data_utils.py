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