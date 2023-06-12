import argparse
import os
import json
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
from scipy.stats import pearsonr

import utils.globals as uglobals
from models.deberta_for_eval import DebertaForEval
from utils.finetune_data_utils import make_finetuning_loader_bleurt, get_concordant_discordant

def run(args):
    if args.debug:
        args.stage = 'finetune_simpeval'
        args.batch_size = 3
        args.batch_size_dev = 3
        args.n_epoch = 2

    print(args)

    # Device
    torch.manual_seed(21)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Setup Tensorboard
    date_str = str(datetime.datetime.now())[:-7].replace(':','-')
    writer = SummaryWriter(log_dir=f'{uglobals.RESULTS_DIR}/runs/{args.name}/batch_size={args.batch_size}, Adam_lr={args.lr}/{date_str}' ,comment=args)

    # Training setup
    checkpoint='lucadiliello/BLEURT-20-D12'
    model = BleurtForSequenceClassification.from_pretrained(checkpoint) 
    model.to(device)
    # Reinitialize the regression head
    torch.nn.init.normal_(model.classifier.weight)
    torch.nn.init.normal_(model.classifier.bias)
    tokenizer = BleurtTokenizer.from_pretrained(checkpoint)

    criterion = torch.nn.MSELoss()

    optimizer_params = model.parameters()
    optimizer = AdamW(optimizer_params, lr=args.lr)

     # Load checkpoint
    if args.checkpoint != '':
        print(f'loading checkpoint: {args.checkpoint}')
        model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict'])
        optimizer.load_state_dict(torch.load(args.checkpoint, map_location=device)['optimizer_state_dict'])

    # Data loaders for the current stage
    eval_n_epoch = 1
    if args.stage == 'finetune_simpeval':
        train_loader = make_finetuning_loader_bleurt(f'{uglobals.STAGE3_PROCESSED_DIR}/simpeval_asset_train.csv', args.batch_size)
        dev_loader = make_finetuning_loader_bleurt(f'{uglobals.STAGE3_PROCESSED_DIR}/simpeval_asset_dev.csv', args.batch_size_dev, shuffle=False)
    elif args.stage == 'finetune_simpda':
        pass
    else:
        raise NotImplementedError
    if args.save_epoch > 0:
        eval_n_epoch = args.save_epoch

    if args.stage == 'finetune_simpeval':
        n_iter = 0
        n_prev_iter = 0
        running_loss = 0
        for epoch in range(args.n_epoch):
            # Train
            for batch_idx, batch in enumerate(train_loader):
                if args.debug and batch_idx > 3:
                    break
                loss = train_step(batch, model, tokenizer, optimizer, criterion, device)
                n_iter += 1
                writer.add_scalar('Loss/train_batch', loss, n_iter)
                running_loss += loss.detach()

            # Batch loss
            print(f'Epoch {epoch} done. Loss: {running_loss/(n_iter-n_prev_iter)}')
            writer.add_scalar('Loss/train_avg', running_loss/(n_iter-n_prev_iter), n_iter)
            n_prev_iter = n_iter
            running_loss = 0

            if epoch % eval_n_epoch == 0:
                # Eval
                dev_loss = 0
                for batch_idx, batch in enumerate(dev_loader):
                    if args.debug and batch_idx > 3:
                        break
                    
                    dev_loss_iter = eval_step(batch, model, tokenizer, criterion, device)
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
                
    else:
        for measure in ['adequacy', 'fluency', 'simplicity']:
            pearsons = []
            kendall_likes = []
            for fold_idx in range(4):
                # Get loaders
                train_loader = make_finetuning_loader_bleurt(f'{uglobals.STAGE3_PROCESSED_DIR}/simp_da_fold{fold_idx}_train_{measure}.csv', args.batch_size)
                dev_loader = make_finetuning_loader_bleurt(f'{uglobals.STAGE3_PROCESSED_DIR}/simp_da_fold{fold_idx}_dev_{measure}.csv', args.batch_size_dev, shuffle=False)
                test_loader = make_finetuning_loader_bleurt(f'{uglobals.STAGE3_PROCESSED_DIR}/simp_da_fold{fold_idx}_test_{measure}.csv', args.batch_size_dev, shuffle=False)
                
                # Training setup
                checkpoint='lucadiliello/BLEURT-20-D12'
                model = BleurtForSequenceClassification.from_pretrained(checkpoint) 
                model.to(device)
                # Reinitialize the regression head
                torch.nn.init.normal_(model.classifier.weight)
                torch.nn.init.normal_(model.classifier.bias)
                tokenizer = BleurtTokenizer.from_pretrained(checkpoint)

                criterion = torch.nn.MSELoss()

                optimizer_params = model.parameters()
                optimizer = AdamW(optimizer_params, lr=args.lr)

                best_dev_loss = 100000
                best_dev_cp = ''
                
                n_iter = 0
                n_prev_iter = 0
                running_loss = 0

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

                    if epoch % eval_n_epoch == 0:
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
                        if dev_loss < best_dev_loss:
                            best_dev_loss = dev_loss
                            best_dev_cp = save_dir
                        
                # Eval
                print(f'Loading the checkpoint with the best dev loss: {best_dev_cp}')
                model.load_state_dict(torch.load(best_dev_cp, map_location=device)['model_state_dict'])
                preds = []
                for batch_idx, batch in enumerate(test_loader):
                    if args.debug and batch_idx > 3:
                        break

                    pred = infer_step(batch, model, tokenizer, criterion, device, args)
                    preds = preds + pred

                human_scores = test_loader.dataset.df['score'].tolist()

                # Pearson Corrlation
                pearson = pearsonr(preds, human_scores).statistic
                print(f'Pearson correlation: {pearson}')
                pearsons.append(pearson)

                # kendall tau-like
                kendall = get_concordant_discordant(preds, human_scores)
                print(f'Kendall Tau-like: {kendall}')
                kendall_likes.append(kendall)

            print(measure)
            print('Pearsons:', sum(pearsons) / len(pearsons))
            print('Kendall-Tau-likes:', sum(kendall_likes) / len(kendall_likes))

def train_step(batch, model, tokenizer, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()

    # Unpack the batch
    src, pred, ref, scores = batch
    scores = scores.to(device).float()
    
    # Forward
    try:
        inputs = tokenizer(pred, ref, padding='longest', return_tensors='pt').to(device)
    except:
        print(pred)
        print(ref)
        raise
    pred = model(**inputs).logits.flatten()

    # Loss
    loss = criterion(pred, scores).float()

    # Backward
    loss.backward()
    optimizer.step()

    return loss
    

def eval_step(batch, model, tokenizer, criterion, device):
    model.eval()
    with torch.no_grad():

        # Unpack the batch
        src, pred, ref, scores = batch
        scores = scores.to(device).float()
        
        # Forward
        inputs = tokenizer(pred, ref, padding='longest', return_tensors='pt').to(device)
        pred = model(**inputs).logits.flatten()

        # Loss
        loss = criterion(pred, scores).float()

        return loss
    

def infer_step(batch, model, tokenizer, criterion, device, args):
    model.eval()
    with torch.no_grad():

        # Unpack the batch
        src, pred, ref, scores = batch
        scores = scores.to(device).float()
        
        # Forward
        inputs = tokenizer(pred, ref, padding='longest', return_tensors='pt').to(device)
        pred = model(**inputs).logits.flatten().cpu().tolist()

        return pred

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='unnamed')
    parser.add_argument('--debug', action='store_true')

    # Formulation
    parser.add_argument('--stage', type=str)
    parser.add_argument('--save_epoch', default=0, type=int)

    # Training
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--batch_size_dev', default=8, type=int)
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--n_epoch', default=1000, type=int)
    parser.add_argument('--checkpoint', default='', type=str)

    args = parser.parse_args()

    run(args)