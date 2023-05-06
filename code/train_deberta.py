import argparse
import os
import json
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW

import utils.globals as uglobals
from models.deberta_for_eval import DebertaForEval

def run(args):
    print(args)

    # Device
    torch.manual_seed(21)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Make loaders

    # Setup Tensorboard
    date_str = str(datetime.datetime.now())[:-7].replace(':','-')
    writer = SummaryWriter(log_dir=f'../result/runs/{args.name}/batch_size={args.batch_size}, Adam_lr={args.lr}/{date_str}' ,comment=args)

    # Training setup
    model = DebertaForEval(uglobals.DERBERTA_MODEL_DIR, uglobals.DERBERTA_TOKENIZER_DIR, device)
    criterion = torch.nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)

     # Load checkpoint
    if args.checkpoint != '':
        print(f'loading checkpoint: {args.checkpoint}')
        model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict'])
        optimizer.load_state_dict(torch.load(args.checkpoint, map_location=device)['optimizer_state_dict'])




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='unnamed')

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--batch_size_dev', default=8, type=int)
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--n_epoch', default=1000, type=int)
    parser.add_argument('--checkpoint', default='', type=str)

    args = parser.parse_args()

    run(args)