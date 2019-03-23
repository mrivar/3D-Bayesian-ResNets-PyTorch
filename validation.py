import torch
from torch.autograd import Variable
import time
import sys

from utils import AverageMeter, calculate_accuracy


def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        with torch.no_grad():
            inputs = Variable(inputs)#, volatile=True)
            targets = Variable(targets)#, volatile=True)

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

        try: losses.update(loss.data[0], inputs.size(0))
        except: losses.update(loss.data.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

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
                  acc=accuracies))

    logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

    return losses.avg
