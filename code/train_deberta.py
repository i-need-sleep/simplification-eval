import argparse
import os
import json
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW

import utils.globals as uglobals
from models.deberta_for_eval import DebertaForEval
from utils.pretrain_stage_1_data_utils import make_pretraining_loader

def run(args):
    if args.debug:
        args.stage = 'pretrain_1'
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
        optimizer.load_state_dict(torch.load(args.checkpoint, map_location=device)['optimizer_state_dict'])

    # Data loaders for the current stage
    if args.stage == 'pretrain_1':
        train_loader = make_pretraining_loader(f'{uglobals.PROCESSED_DIR}/openwebtext/train/train.csv', model.tokenizer, args.batch_size)
        dev_loader = make_pretraining_loader(f'{uglobals.PROCESSED_DIR}/openwebtext/train/dev.csv', model.tokenizer, args.batch_size_dev)
    else:
        raise NotImplementedError

    n_iter = 0
    n_prev_iter = 0
    running_loss = 0
    best_dev_loss = 0
    for epoch in range(args.n_epoch):
        # Train
        for batch_idx, batch in enumerate(train_loader):
            if args.debug and batch_idx > 3:
                break
            loss = train_step(batch, model, optimizer, criterion, device)
            n_iter += 1
            writer.add_scalar('Loss/train_batch', loss, n_iter)
            running_loss += loss.detach()

        # Batch loss
        print(f'Epoch {epoch} done. Loss: {running_loss/(n_iter-n_prev_iter)}')
        writer.add_scalar('Loss/train_avg', running_loss/(n_iter-n_prev_iter), n_iter)
        n_prev_iter = n_iter
        running_loss = 0

        # Eval
        dev_loss = 0
        for batch_idx, batch in enumerate(dev_loader):
            if args.debug and batch_idx > 3:
                break
            
            dev_loss_iter = eval_step(batch, model, criterion, device)
            dev_loss += dev_loss_iter.detach()

        dev_loss = dev_loss / len(dev_loader)
        print(f'Dev loss: {dev_loss}')
        writer.add_scalar('loss/dev', dev_loss, n_iter)

        # Save
        try:
            os.makedirs(f'{uglobals.CHECKPOINTS_DIR}/{args.name}')
        except:
            pass
        save_dir = f'{uglobals.CHECKPOINTS_DIR}/{args.name}/lr{args.lr}_{epoch}_{batch_idx}_{dev_loss}.bin'
        print(f'Saving at: {save_dir}')
        torch.save({
            'epoch': epoch,
            'step': n_iter,
            'model_state_dict': model.state_dict(),
            'optimizer_bert_state_dict': optimizer.state_dict(),
            }, save_dir)

def train_step(batch, model, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()

    # Unpack the batch
    sents, scores, score_masks = batch
    scores = scores.to(device)
    score_masks = score_masks.to(device)
    
    # Forward
    pred = model(sents)

    # Apply score masks: zero-out unavailable scores
    pred = pred * score_masks
    scores = scores * score_masks

    # Loss
    loss = criterion(pred, scores)

    # Backward
    loss.backward()
    optimizer.step()

    return loss
    

def eval_step(batch, model, criterion, device):
    model.eval()
    with torch.no_grad():

        # Unpack the batch
        sents, scores, score_masks = batch
        scores = scores.to(device)
        score_masks = score_masks.to(device)
        
        # Forward
        pred = model(sents)

        # Apply score masks: zero-out unavailable scores
        pred = pred * score_masks
        scores = scores * score_masks

        # Loss
        loss = criterion(pred, scores)

        return loss
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='unnamed')
    parser.add_argument('--debug', action='store_true')

    # Formulation
    parser.add_argument('--stage', type=str)

    # Training
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--batch_size_dev', default=16, type=int)
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--n_epoch', default=1000, type=int)
    parser.add_argument('--checkpoint', default='', type=str)

    args = parser.parse_args()

    run(args)