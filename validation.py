import torch
from torch.autograd import Variable
from torch.nn.functional import softmax
import numpy as np
import time
import sys
import math

from utils import AverageMeter, calculate_test_accuracy
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def val_epoch(epoch, data_loader, model, criterion, opt, logger, uncertainty_logger):
    #print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_logpy = AverageMeter()
    accuracies = AverageMeter()
    accuracies_mean = AverageMeter()
    accuracies_vote = AverageMeter()
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

            outputs = torch.tensor([]).to(DEVICE)
            for _ in range(opt.num_samples):
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
                    outputs_aux, kl = model(inputs)
                    loss, logpy = criterion(outputs_aux, targets, kl, beta)
                    losses_logpy.update(logpy.data.detach().item(), inputs.size(0))
                else:
                    outputs_aux = model(inputs)
                    loss = criterion(outputs_aux, targets)
                outputs = torch.cat((outputs, outputs_aux), 0)
                losses.update(loss.data.detach().item(), inputs.size(0))

        acc, acc_mean, acc_vote, res = calculate_test_accuracy(softmax(outputs, dim=1), targets, targets.repeat(opt.num_samples), opt)
        accuracies.update(acc, inputs.size(0))
        accuracies_mean.update(acc_mean, inputs.size(0))
        accuracies_vote.update(acc_vote, inputs.size(0))

        conf.append(res)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('| Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.avg:.3f} ({batch_time.val:.3f})\t'
              'Data {data_time.avg:.3f} ({data_time.val:.3f})\t'
              'Loss {loss.avg:.4f} ({loss.val:.4f})\t'
              'Acc {acc.avg:.3f} ({acc.val:.3f})\t'
              'AccMean {acc_mean.avg:.3f} ({acc_mean.val:.3f})\t'
              'AccVote {acc_vote.avg:.3f} ({acc_vote.val:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies,
                  acc_mean=accuracies_mean,
                  acc_vote=accuracies_vote), end='\r')
    print()
    # Calculate aleatoric and epistemic uncertainty
    p_hat = np.array(conf)
    epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2
    aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)

    # Get random parameters mean and deviation
    random_param_mean,random_param_log_alpha,total_param_mean,total_param_log_alpha  = 0, 0, 0, 0
    for k,v in model.named_parameters():
      if k == "module.layer1.1.conv1.qw_mu":
        random_param_mean = v[0][0][0][0][0].item()
        total_param_mean = v.mean().item()
      if k == "module.layer1.1.conv1.qw_log_alpha":
        random_param_log_alpha = v[0][0][0][0][0].item()
        total_param_log_alpha = v.mean().item()


    logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg,
      'acc_mean': accuracies_mean.avg, 'acc_vote': accuracies_vote.avg})
    uncertainty_logger.log({'epoch': epoch,
      'epistemic': epistemic, 'aleatoric': aleatoric,
      'random_param_mean': random_param_mean, 'random_param_log_alpha': random_param_log_alpha,
      'total_param_mean': total_param_mean, 'total_param_log_alpha': total_param_log_alpha})

    if opt.bayesian:
      return losses_logpy.avg
    return losses.avg
