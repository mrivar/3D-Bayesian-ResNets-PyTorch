import torch
from torch.autograd import Variable
from torch.nn.functional import softmax
import numpy as np
import time
import sys
import math

from utils import AverageMeter, calculate_test_accuracy


def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    conf = []
    m = math.ceil(len(data_loader) / opt.batch_size)

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
        acc, res = calculate_test_accuracy(softmax(outputs), targets)

        try: losses.update(loss.data[0], inputs.size(0))
        except: losses.update(loss.data.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        conf.append(res)

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

    # Calculate aleatoric and epistemic uncertainty
    p_hat = np.array(conf)
    epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2
    aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)

    # Get random parameters mean and deviation
    random_param_mean   = 0
    random_param_logvar = 0
    for k,v in model.named_parameters():
      if k == "module.layer1.1.conv1.qw_mean":
        random_param_mean = v[0][0][0][0][0].item()
      if k == "module.layer1.1.conv1.qw_logvar":
        random_param_logvar = v[0][0][0][0][0].item()


    logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg,
      'epistemic': epistemic, 'aleatoric': aleatoric,
      'random_param_mean': random_param_mean, 'random_param_logvar': random_param_logvar})

    return losses.avg
