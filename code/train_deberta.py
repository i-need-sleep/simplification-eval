import argparse
import os
import json
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
from scipy.stats import pearsonr
import numpy as np

import utils.globals as uglobals
from models.deberta_for_eval import DebertaForEval
from utils.pretrain_stage_1_data_utils import make_pretraining_loader
from utils.pretrain_stage_2_data_utils import make_pretraining_stage2_loader
from utils.finetune_data_utils import make_finetuning_loader, get_concordant_discordant
from infer_deberta import infer_step

def run(args):
    print(args)

    # Device
    # torch.manual_seed(21)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Setup Tensorboard
    date_str = str(datetime.datetime.now())[:-7].replace(':','-')
    writer = SummaryWriter(log_dir=f'{uglobals.RESULTS_DIR}/runs/{args.name}/batch_size={args.batch_size}, Adam_lr={args.lr}/{date_str}' ,comment=args)

    # Training setup
    if args.backbone == 'deberta':
        model = DebertaForEval(uglobals.DERBERTA_MODEL_DIR, uglobals.DERBERTA_TOKENIZER_DIR, device, head_type=args.head_type)
    elif args.backbone == 'roberta':
        model = DebertaForEval(uglobals.RORBERTA_MODEL_DIR, uglobals.RORBERTA_TOKENIZER_DIR, device, head_type=args.head_type, backbone='roberta')
    else:
        raise NotImplementedError
    criterion = torch.nn.MSELoss()

    optimizer_params = model.parameters()
    optimizer = AdamW(optimizer_params, lr=args.lr)

    # Load checkpoint
    if args.checkpoint != '':
        print(f'loading checkpoint: {args.checkpoint}')
        model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict'])

    # Ablation on first-stage pretraining supervision signals
    if args.ablate_supervision == 'none':
        ablation_mask = [1 for _ in 13]
    elif args.ablate_supervision == 'meaning':
        ablation_mask = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    elif args.ablate_supervision == 'fluency':
        ablation_mask = [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    elif args.ablate_supervision == 'simplicity':
        ablation_mask = [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
    else:
        raise NotImplementedError

    # Data loaders for the current stage
    eval_n_epoch = 1
    if args.stage == 'pretrain_1':
        train_loader = make_pretraining_loader(f'{uglobals.PROCESSED_DIR}/openwebtext/train/train.csv', model.tokenizer, args.batch_size)
        dev_loader = make_pretraining_loader(f'{uglobals.PROCESSED_DIR}/openwebtext/train/dev.csv', model.tokenizer, args.batch_size_dev, shuffle=False)
        eval_n_epoch = 1
    elif args.stage == 'pretrain_2':
        train_loader = make_pretraining_stage2_loader(f'{uglobals.STAGE2_OUTPUTS_DIR}/train/train.csv', model.tokenizer, args.batch_size)
        dev_loader = make_pretraining_stage2_loader(f'{uglobals.STAGE2_OUTPUTS_DIR}/train/dev.csv', model.tokenizer, args.batch_size_dev, shuffle=False)
        eval_n_epoch = 4
    elif args.stage == 'finetune_simpeval':
        train_loader = make_finetuning_loader(f'{uglobals.STAGE3_PROCESSED_DIR}/simpeval_asset_train.csv', model.tokenizer, args.batch_size)
        dev_loader = make_finetuning_loader(f'{uglobals.STAGE3_PROCESSED_DIR}/simpeval_asset_dev.csv', model.tokenizer, args.batch_size_dev, shuffle=False)
    elif args.stage in ['finetune_simpda']:
        pass
    #     train_loader = make_finetuning_loader(f'{uglobals.STAGE3_PROCESSED_DIR}/simp_da_train_simplicity.csv', model.tokenizer, args.batch_size)
    #     dev_loader = make_finetuning_loader(f'{uglobals.STAGE3_PROCESSED_DIR}/simp_da_dev_simplicity.csv', model.tokenizer, args.batch_size_dev, shuffle=False)
    else:
        raise NotImplementedError
    if args.save_epoch > 0:
        eval_n_epoch = args.save_epoch

    if not args.stage in ['finetune_simpda']:
        n_iter = 0
        n_prev_iter = 0
        running_loss = 0
        for epoch in range(args.n_epoch):
            # Train
            for batch_idx, batch in enumerate(train_loader):
                if args.debug and batch_idx > 3:
                    break
                loss = train_step(batch, model, optimizer, criterion, device, ablation_mask)
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
                    
                    dev_loss_iter = eval_step(batch, model, criterion, device, ablation_mask)
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
                
    # K-fold cross validation
    else:
        for measure in ['adequacy', 'fluency', 'simplicity']:
            print(measure)
            pearsons = []
            kendall_likes = []
            for fold_idx in range(4):
                # Get loaders
                train_loader = make_finetuning_loader(f'{uglobals.STAGE3_PROCESSED_DIR}/simp_da_fold{fold_idx}_train_{measure}.csv', model.tokenizer, args.batch_size)
                dev_loader = make_finetuning_loader(f'{uglobals.STAGE3_PROCESSED_DIR}/simp_da_fold{fold_idx}_dev_{measure}.csv', model.tokenizer, args.batch_size_dev, shuffle=False)
                test_loader = make_finetuning_loader(f'{uglobals.STAGE3_PROCESSED_DIR}/simp_da_fold{fold_idx}_test_{measure}.csv', model.tokenizer, args.batch_size_dev, shuffle=False)
                
                # Training setup
                model = DebertaForEval(uglobals.DERBERTA_MODEL_DIR, uglobals.DERBERTA_TOKENIZER_DIR, device, head_type=args.head_type)
                criterion = torch.nn.MSELoss()


                optimizer_params = model.parameters()
                optimizer = AdamW(optimizer_params, lr=args.lr)

                # Load checkpoint
                if args.checkpoint != '':
                    print(f'loading checkpoint: {args.checkpoint}')
                    model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict'])
                    if args.cont_training:
                        optimizer.load_state_dict(torch.load(args.checkpoint, map_location=device)['optimizer_bert_state_dict'])

                best_dev_loss = 100000
                best_dev_cp = ''
                
                n_iter = 0
                n_prev_iter = 0
                running_loss = 0

                n_iter = 0
                n_prev_iter = 0
                running_loss = 0

                for epoch in range(args.n_epoch):

                    # Train
                    for batch_idx, batch in enumerate(train_loader):
                        if args.debug and batch_idx > 3:
                            break
                        loss = train_step(batch, model, optimizer, criterion, device, ablation_mask)
                        n_iter += 1
                        writer.add_scalar('Loss/train_batch', loss, n_iter)
                        running_loss += loss.detach()

                    # Batch loss
                    # print(f'Epoch {epoch} done. Loss: {running_loss/(n_iter-n_prev_iter)}')
                    writer.add_scalar('Loss/train_avg', running_loss/(n_iter-n_prev_iter), n_iter)
                    n_prev_iter = n_iter
                    running_loss = 0

                    if epoch % eval_n_epoch == 0:
                        # Eval
                        dev_loss = 0
                        for batch_idx, batch in enumerate(dev_loader):
                            if args.debug and batch_idx > 3:
                                break
                            
                            dev_loss_iter = eval_step(batch, model, criterion, device, ablation_mask)
                            dev_loss += dev_loss_iter.detach()

                        dev_loss = dev_loss / len(dev_loader)
                        # print(f'Dev loss: {dev_loss}')
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

                    pred = infer_step(batch, model, criterion, device, args)
                    preds = preds + pred

                human_scores = test_loader.dataset.df['score'].tolist()

                # Pearson Corrlation
                pearson = pearsonr(preds, human_scores).statistic
                pearsons.append(pearson)

                # kendall tau-like
                kendall = get_concordant_discordant(preds, human_scores)
                kendall_likes.append(kendall)

            print(measure)
            print('Pearsons:', sum(pearsons) / len(pearsons))
            print(pearsons)
            pearsons = np.array(pearsons)
            print('std:', np.std(pearsons))
            print('Kendall-Tau-likes:', sum(kendall_likes) / len(kendall_likes))
            print(kendall_likes)
            kendall_likes = np.array(kendall_likes)
            print('std:', np.std(kendall_likes))

def train_step(batch, model, optimizer, criterion, device, ablation_mask):
    model.train()
    optimizer.zero_grad()

    # Unpack the batch
    sents, scores, score_masks = batch
    scores = scores.to(device).float()
    score_masks = score_masks.to(device).float()

    score_masks = score_masks * torch.tensor(ablation_mask).to(device)
    
    # Forward
    pred = model(sents).float()

    # Apply score masks: zero-out unavailable scores
    pred = pred * score_masks
    scores = scores * score_masks

    # Loss
    loss = criterion(pred, scores).float()

    # Backward
    loss.backward()
    optimizer.step()

    return loss
    

def eval_step(batch, model, criterion, device, ablation_mask):
    model.eval()
    with torch.no_grad():

        # Unpack the batch
        sents, scores, score_masks = batch
        scores = scores.to(device)
        score_masks = score_masks.to(device)

        score_masks = score_masks * torch.tensor(ablation_mask).to(device)
        
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
    parser.add_argument('--backbone', type=str) # debeta, roberta
    parser.add_argument('--head_type', default='mlp', type=str)
    parser.add_argument('--save_epoch', default=0, type=int)

    # Training
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--batch_size_dev', default=16, type=int)
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--n_epoch', default=1000, type=int)
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--cont_training', action='store_true')
    parser.add_argument('--freeze_deberta', action='store_true')
    parser.add_argument('--ablate_supervision', default='none', type=str) # none, meaning, fluency, simplicity

    args = parser.parse_args()
    
    if args.debug:
        args.stage = 'pretrain_1'
        args.backbone = 'roberta'
        args.batch_size = 3
        args.batch_size_dev = 3
        args.n_epoch = 1
        args.head_type = 'linear'
        args.ablate_supervision = 'meaning'

    run(args)