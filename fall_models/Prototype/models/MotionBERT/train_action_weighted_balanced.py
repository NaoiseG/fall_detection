import os
import numpy as np
import time
import sys
import argparse
import pickle
import errno
from collections import OrderedDict
import tensorboardX
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from lib.utils.tools import *
from lib.utils.learning import *
from lib.model.loss import *
from lib.data.dataset_action import NTURGBD
from lib.model.model_action import ActionNet

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-freq', '--print_freq', default=100)
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    opts = parser.parse_args()
    return opts


# -------------------------------------------------------------------------
# UPFall / imbalanced-class tweaks
# -------------------------------------------------------------------------
# Minimal toggles: flip these two constants if you want to revert behaviour.
USE_CLASS_WEIGHTS = True          # If True, use weighted CrossEntropyLoss
BEST_ON_BALANCED_ACC = True       # If True, save best checkpoint by balanced accuracy (mean recall)

def compute_class_weights_from_pkl(data_path: str, train_split: str, num_classes: int,
                                   mode: str = "inv_sqrt", eps: float = 1e-6) -> torch.Tensor:
    """
    Compute per-class weights from the training split of a MotionBERT action .pkl.

    data_path: e.g. 'data/action/upfall.pkl'
    train_split: e.g. 'xsub_train'
    num_classes: args.action_classes

    mode:
      - 'inv'      : weight ~ 1 / count
      - 'inv_sqrt' : weight ~ 1 / sqrt(count)   (usually more stable)
    Returns a float32 tensor of shape (num_classes,), normalized to mean=1.
    """
    with open(data_path, "rb") as f:
        ds = pickle.load(f)

    split_key = train_split
    if split_key not in ds.get("split", {}):
        raise KeyError(f"Split '{split_key}' not found in {data_path}. Available: {list(ds.get('split', {}).keys())}")

    train_dirs = set(ds["split"][split_key])
    counts = np.zeros((num_classes,), dtype=np.int64)

    # Count labels in the train split only
    for ann in ds.get("annotations", []):
        fd = ann.get("frame_dir", None)
        if fd in train_dirs:
            lab = int(ann["label"])
            if 0 <= lab < num_classes:
                counts[lab] += 1

    # Avoid divide-by-zero for missing classes (still keep a finite weight)
    safe = np.maximum(counts, 1)

    if mode == "inv":
        w = 1.0 / (safe.astype(np.float64) + eps)
    else:
        # default: inv_sqrt
        w = 1.0 / (np.sqrt(safe.astype(np.float64)) + eps)

    # Normalize so average weight is 1 (keeps loss scale roughly comparable)
    w = w / (np.mean(w) + eps)

    print("\n[Class weighting] Train label counts:", {i: int(c) for i, c in enumerate(counts) if c > 0})
    print("[Class weighting] Example weights (first 11):", [float(x) for x in w[:min(11, len(w))]])
    return torch.tensor(w, dtype=torch.float32)


def confusion_metrics_from_cm(cm: np.ndarray, eps: float = 1e-12):
    """
    Compute balanced accuracy (mean recall over classes present) and macro-F1 from a confusion matrix.
    cm shape: (C,C) with rows=gt, cols=pred
    """
    cm = cm.astype(np.float64)
    support = cm.sum(axis=1)  # gt count per class
    valid = support > 0

    tp = np.diag(cm)
    recall = tp / (support + eps)

    pred_support = cm.sum(axis=0)
    precision = tp / (pred_support + eps)

    f1 = 2 * precision * recall / (precision + recall + eps)

    balanced_acc = float(np.mean(recall[valid])) if np.any(valid) else 0.0
    macro_f1 = float(np.mean(f1[valid])) if np.any(valid) else 0.0
    return balanced_acc, macro_f1, recall, precision, f1

def validate(test_loader, model, criterion, num_classes: int):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        end = time.time()
        for idx, (batch_input, batch_gt) in tqdm(enumerate(test_loader)):
            batch_size = len(batch_input)    
            if torch.cuda.is_available():
                batch_gt = batch_gt.cuda()
                batch_input = batch_input.cuda()
            output = model(batch_input)    # (N, num_classes)
            loss = criterion(output, batch_gt)
            # accumulate confusion matrix for balanced metrics
            pred = torch.argmax(output, dim=1)
            gt_cpu = batch_gt.detach().cpu().numpy().astype(np.int64)
            pr_cpu = pred.detach().cpu().numpy().astype(np.int64)
            np.add.at(cm, (gt_cpu, pr_cpu), 1)

            # update metric
            losses.update(loss.item(), batch_size)
            acc1, acc5 = accuracy(output, batch_gt, topk=(1, 5))
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx+1) % opts.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                       idx, len(test_loader), batch_time=batch_time,
                       loss=losses, top1=top1, top5=top5))
        # balanced metrics (more informative than total accuracy under class imbalance)
    balanced_acc, macro_f1, recall, precision, f1 = confusion_metrics_from_cm(cm)
    # Print a quick per-class recall summary (helps diagnose minority/fall classes)
    support = cm.sum(axis=1)
    valid = support > 0
    if np.any(valid):
        # show 5 worst recalls among classes that appear in this split
        rec_valid = recall.copy()
        rec_valid[~valid] = 2.0  # push absent classes to the end
        worst = np.argsort(rec_valid)[:min(5, num_classes)]
        worst_str = ', '.join([f"{int(i)}:{float(recall[i]):.2f}({int(support[i])})" for i in worst])
        print(f"Worst recall (class:recall(support)): {worst_str}")
    print(f"Test summary: Acc@1 {float(top1.avg):.3f} | BalancedAcc {balanced_acc:.3f} | MacroF1 {macro_f1:.3f}")
    return losses.avg, top1.avg, top5.avg, balanced_acc, macro_f1, recall



def train_with_config(args, opts):
    print(args)
    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))
    model_backbone = load_backbone(args)
    if args.finetune:
        if opts.resume or opts.evaluate:
            pass
        else:
            chk_filename = os.path.join(opts.pretrained, opts.selection)
            print('Loading backbone', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)['model_pos']
            model_backbone = load_pretrained_weights(model_backbone, checkpoint)
    if args.partial_train:
        model_backbone = partial_train_layers(model_backbone, args.partial_train)
    model = ActionNet(backbone=model_backbone, dim_rep=args.dim_rep, num_classes=args.action_classes, dropout_ratio=args.dropout_ratio, version=args.model_version, hidden_dim=args.hidden_dim, num_joints=args.num_joints)
    # criterion is created later (after optional class-weight computation)
    criterion = None
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        # criterion is moved to CUDA after it is created

    best_acc = 0
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)
    print('Loading dataset...')
    trainloader_params = {
          'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 8,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    testloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 8,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    data_path = 'data/action/%s.pkl' % args.dataset
    ntu60_xsub_train = NTURGBD(data_path=data_path, data_split=args.data_split+'_train', n_frames=args.clip_len, random_move=args.random_move, scale_range=args.scale_range_train)
    ntu60_xsub_val = NTURGBD(data_path=data_path, data_split=args.data_split+'_val', n_frames=args.clip_len, random_move=False, scale_range=args.scale_range_test)

    # ------------------------------------------------------------------
    # Optional: weighted CE loss to upweight minority classes.
    # This reads the .pkl once and derives weights from the TRAIN split.
    # ------------------------------------------------------------------
    class_weights = None
    if USE_CLASS_WEIGHTS:
        # args.data_split is usually 'xsub' so train split key is 'xsub_train'
        train_split_key = args.data_split + '_train'
        class_weights = compute_class_weights_from_pkl(data_path, train_split_key, args.action_classes, mode='inv_sqrt')

    train_loader = DataLoader(ntu60_xsub_train, **trainloader_params)
    test_loader = DataLoader(ntu60_xsub_val, **testloader_params)

    # Recreate criterion with weights (if enabled)
    if class_weights is not None:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        if class_weights is not None:
            class_weights = class_weights.cuda()
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        criterion = criterion.cuda()
        
    checkpoint = None

    # Auto-resume convenience: if no explicit --resume/--evaluate is provided and a
    # latest checkpoint exists under --checkpoint, resume from it.
    # Guard: if the checkpoint was trained with a different number of classes,
    # do NOT attempt to resume (the classifier head shape will mismatch).
    auto_chk = os.path.join(opts.checkpoint, "latest_epoch.bin")
    if (not opts.resume) and (not opts.evaluate) and os.path.exists(auto_chk):
        try:
            checkpoint = torch.load(auto_chk, map_location=lambda storage, loc: storage)
            state = checkpoint.get("model", checkpoint)
            head_w = None
            if isinstance(state, dict):
                for k in ("module.head.fc2.weight", "head.fc2.weight"):
                    if k in state:
                        head_w = state[k]
                        break
            if head_w is not None:
                out_dim = int(head_w.shape[0])
                if out_dim != int(args.action_classes):
                    raise RuntimeError(
                        f"Found existing checkpoint '{auto_chk}' with classifier out_dim={out_dim}, "
                        f"but config expects action_classes={int(args.action_classes)}. "
                        f"Use a new --checkpoint directory for this run or delete the old checkpoint to start fresh."
                    )
            opts.resume = auto_chk
        except RuntimeError:
            raise
        except Exception:
            # If we can't inspect the checkpoint for compatibility, fall back to the old behaviour
            # (resume and let load_state_dict surface any issues).
            checkpoint = None
            opts.resume = auto_chk

    if opts.resume or opts.evaluate:
        chk_filename = opts.evaluate if opts.evaluate else opts.resume
        print('Loading checkpoint', chk_filename)
        if checkpoint is None or chk_filename != auto_chk:
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'], strict=True)
    
    if not opts.evaluate:
        optimizer = optim.AdamW(
            [     {"params": filter(lambda p: p.requires_grad, model.module.backbone.parameters()), "lr": args.lr_backbone},
                  {"params": filter(lambda p: p.requires_grad, model.module.head.parameters()), "lr": args.lr_head},
            ],      lr=args.lr_backbone, 
                    weight_decay=args.weight_decay
        )

        scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay)
        st = 0
        print('INFO: Training on {} batches'.format(len(train_loader)))
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
            lr = checkpoint['lr']
            if 'best_acc' in checkpoint and checkpoint['best_acc'] is not None:
                best_acc = checkpoint['best_acc']
        # Training
        for epoch in range(st, args.epochs):
            print('Training epoch %d.' % epoch)
            losses_train = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            model.train()
            end = time.time()
            iters = len(train_loader)
            for idx, (batch_input, batch_gt) in tqdm(enumerate(train_loader)):    # (N, 2, T, 17, 3)
                data_time.update(time.time() - end)
                batch_size = len(batch_input)
                if torch.cuda.is_available():
                    batch_gt = batch_gt.cuda()
                    batch_input = batch_input.cuda()
                output = model(batch_input) # (N, num_classes)
                optimizer.zero_grad()
                loss_train = criterion(output, batch_gt)
                losses_train.update(loss_train.item(), batch_size)
                acc1, acc5 = accuracy(output, batch_gt, topk=(1, 5))
                top1.update(acc1[0], batch_size)
                top5.update(acc5[0], batch_size)
                loss_train.backward()
                optimizer.step()    
                batch_time.update(time.time() - end)
                end = time.time()
            if (idx + 1) % opts.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       epoch, idx + 1, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses_train, top1=top1))
                sys.stdout.flush()
                
            test_loss, test_top1, test_top5, test_bal_acc, test_macro_f1, test_recall = validate(test_loader, model, criterion, args.action_classes)
                
            train_writer.add_scalar('train_loss', losses_train.avg, epoch + 1)
            train_writer.add_scalar('train_top1', top1.avg, epoch + 1)
            train_writer.add_scalar('train_top5', top5.avg, epoch + 1)
            train_writer.add_scalar('test_loss', test_loss, epoch + 1)
            train_writer.add_scalar('test_top1', test_top1, epoch + 1)
            train_writer.add_scalar('test_top5', test_top5, epoch + 1)
            train_writer.add_scalar('test_balanced_acc', test_bal_acc, epoch + 1)
            train_writer.add_scalar('test_macro_f1', test_macro_f1, epoch + 1)
            
            scheduler.step()

            # Save latest checkpoint.
            chk_path = os.path.join(opts.checkpoint, 'latest_epoch.bin')
            print('Saving checkpoint to', chk_path)
            torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_acc' : best_acc
            }, chk_path)

            # Save best checkpoint.
            best_chk_path = os.path.join(opts.checkpoint, 'best_epoch.bin'.format(epoch))
            # Save best checkpoint based on balanced accuracy (mean recall) if enabled.
            score = test_bal_acc if BEST_ON_BALANCED_ACC else float(test_top1)
            if score > best_acc:
                best_acc = score
                print("save best checkpoint")
                torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_acc' : best_acc
                }, best_chk_path)

    if opts.evaluate:
        test_loss, test_top1, test_top5, test_bal_acc, test_macro_f1, test_recall = validate(test_loader, model, criterion, args.action_classes)
        print('Loss {loss:.4f} \t'
              'Acc@1 {top1:.3f} \t'
              'Acc@5 {top5:.3f} \t'.format(loss=test_loss, top1=test_top1, top5=test_top5))

if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    train_with_config(args, opts)
