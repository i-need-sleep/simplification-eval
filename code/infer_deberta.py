import argparse
import os
import json
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
from scipy.stats import pearsonr

import utils.globals as uglobals
from models.deberta_for_eval import DebertaForEval
from utils.pretrain_stage_1_data_utils import make_pretraining_loader
from utils.pretrain_stage_2_data_utils import make_pretraining_stage2_loader
from utils.finetune_data_utils import make_finetuning_loader, get_concordant_discordant

def run(args):
    if args.debug:
        args.stage = 'pretrain_2'
        args.batch_size = 3
        args.batch_size_dev = 3
        args.n_epoch = 1

    print(args)

    # Device
    torch.manual_seed(21)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Setup Tensorboard
    date_str = str(datetime.datetime.now())[:-7].replace(':','-')
    writer = SummaryWriter(log_dir=f'{uglobals.RESULTS_DIR}/runs/{args.name}/batch_size={args.batch_size}, Adam_lr={args.lr}/{date_str}' ,comment=args)

    # Training setup
    model = DebertaForEval(uglobals.DERBERTA_MODEL_DIR, uglobals.DERBERTA_TOKENIZER_DIR, device)
    criterion = torch.nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)

     # Load checkpoint
    if args.checkpoint != '':
        print(f'loading checkpoint: {args.checkpoint}')
        model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict'])
        if args.cont_training:
            optimizer.load_state_dict(torch.load(args.checkpoint, map_location=device)['optimizer_bert_state_dict'])

    # Data loaders for the current stage
    eval_n_epoch = 1
    if args.stage == 'pretrain_1':
        dev_loader = make_pretraining_loader(f'{uglobals.PROCESSED_DIR}/openwebtext/train/dev.csv', model.tokenizer, args.batch_size_dev, shuffle=False)
        eval_n_epoch = 4
    elif args.stage == 'pretrain_2':
        dev_loader = make_pretraining_stage2_loader(f'{uglobals.STAGE2_OUTPUTS_DIR}/train/dev.csv', model.tokenizer, args.batch_size_dev, shuffle=False)
        eval_n_epoch = 4
    elif args.stage == 'finetune_simpeval':
        dev_loader = make_finetuning_loader(f'{uglobals.STAGE3_PROCESSED_DIR}/simpeval_2022.csv', model.tokenizer, args.batch_size_dev, shuffle=False)
    elif args.stage == 'finetune_adequacy':
        dev_loader = make_finetuning_loader(f'{uglobals.STAGE3_PROCESSED_DIR}/simp_da_test_adquacy.csv', model.tokenizer, args.batch_size_dev, shuffle=False)
    elif args.stage == 'finetune_fluency':
        dev_loader = make_finetuning_loader(f'{uglobals.STAGE3_PROCESSED_DIR}/simp_da_test_fluency.csv', model.tokenizer, args.batch_size_dev, shuffle=False)
    elif args.stage == 'finetune_simplicity':
        dev_loader = make_finetuning_loader(f'{uglobals.STAGE3_PROCESSED_DIR}/simp_da_test_simplicity.csv', model.tokenizer, args.batch_size_dev, shuffle=False)
    else:
        raise NotImplementedError

    # Eval
    preds = []
    for batch_idx, batch in enumerate(dev_loader):
        if args.debug and batch_idx > 3:
            break
        
        pred = eval_step(batch, model, criterion, device, args)
        preds = preds + pred

    human_scores = dev_loader.dataset.df['score'].tolist()

    # Pearson Corrlation
    pearson = pearsonr(preds, human_scores).statistic
    print(f'Pearson correlation: {pearson}')

    # kendall tau-like
    kendall = get_concordant_discordant(preds, human_scores)
    print(f'Kendall Tau-like: {kendall}')

    

def eval_step(batch, model, criterion, device, args):
    model.eval()
    with torch.no_grad():

        # Unpack the batch
        sents, scores, score_masks = batch
        scores = scores.to(device)
        score_masks = score_masks.to(device)
        
        # Forward
        pred = model(sents)

        pred = pred[:, -1].reshape(-1).tolist()
        
    return pred

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='unnamed')
    parser.add_argument('--debug', action='store_true')

    # Formulation
    parser.add_argument('--stage', type=str)

    # Training
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--batch_size_dev', default=4, type=int)
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--n_epoch', default=1000, type=int)
    parser.add_argument('--checkpoint', default='../results/checkpoints/simpeval/from_scratch.bin', type=str)
    parser.add_argument('--cont_training', action='store_true')

    args = parser.parse_args()


    for stage in ['simplicity']:
        for model in ['from_scratch', 'from_stage1', 'from_stage2']:
            checkpoint = f'{uglobals.CHECKPOINTS_DIR}/{stage}/{model}.bin'

            args.name = f'{stage}_{model}'
            args.stage = f'finetune_{stage}'
            args.checkpoint = checkpoint
            
            run(args)