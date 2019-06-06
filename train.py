import torch
from torch.autograd import Variable
import time
import os
import sys
import math

from utils import AverageMeter, calculate_accuracy


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    m = math.ceil(len(data_loader) / opt.batch_size)

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        inputs = Variable(inputs)
        targets = Variable(targets)
        if opt.bayesian:
            # Calculate beta
            if opt.beta_type is "Blundell":
                beta = 2 ** (m - (i + 1)) / (2 ** m - 1)
            elif opt.beta_type is "Soenderby":
                beta = min(epoch / (opt.n_epochs // 4), 1)
            elif opt.beta_type is "Standard":
                beta = 1 / m
            else:
                beta = 0
            # Forward Propagation (with KL calc.)
            outputs, kl = model(inputs)
            loss = criterion(outputs, targets, kl, beta)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.data.detach().item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'acc': accuracies.val,
            'lr': optimizer.param_groups[0]['lr']
        })

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies), end="\r")
    print()

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': optimizer.param_groups[0]['lr']
    })

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.checkpoints_path,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)

        # Delete old checkpoint
        delete_file_path = os.path.join(opt.checkpoints_path,
                            'save_{}.pth'.format(
                                epoch - opt.checkpoint * opt.keep_n_checkpoints))

        if os.path.isfile(delete_file_path):
            os.remove(delete_file_path)
