import torch
import csv
import os
from sklearn.metrics import confusion_matrix
import numpy as np
from pandas import DataFrame
from seaborn import heatmap
import pylab
from matplotlib.pyplot import cm as pltcm

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        write_header = True
        #if os.path.isfile(path):
        #    write_header = False
        #self.log_file = open(path, 'a')

        self.log_file = open(path, 'w+')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        if write_header: self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


class ConfusionMatrix(object):

    def __init__(self, n_classes, labels):
        self.cm = np.zeros((n_classes, n_classes))
        self.labels = labels

    def update(self, new_cm):
        try:
            self.cm += new_cm
        except:
            print(self.cm.size())
            print(cm.size())
            self.cm += new_cm

    def normalize(self):
        self.cm = self.cm.astype('float') / self.cm.sum(axis=1)[:, np.newaxis]

    def plot(self, path):
        df_cm = DataFrame(self.cm, index = self.labels, columns = self.labels)
        pylab.figure()
        heatmap(df_cm, annot=True, cmap=pltcm.Blues)
        pylab.savefig(path)
        pylab.clf()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().data.detach().item()

    return 100*n_correct_elems / batch_size

def calculate_test_accuracy(outputs, targets, y, opt):
    batch_size = y.size(0)

    results, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(y.view(1, -1))
    n_correct_elems = correct.float().sum().data.detach().item()
    acc = 100 * n_correct_elems / batch_size

    batch_size = targets.size(0)
    acc_mean, acc_vote = acc, acc
    if opt.bayesian and opt.num_samples > 1:
        # acc_mean
        _, predicted_mean = torch.max(outputs.view(opt.num_samples, batch_size, -1).mean(dim=0).data, 1)
        correct_acc_mean = predicted_mean.eq(targets.data).sum().detach().item()
        acc_mean = 100*correct_acc_mean / batch_size
        # acc_vote
        votes, _ = pred.view(opt.num_samples, -1).mode(dim=0)
        correct_acc_vote = votes.eq(targets.data).sum().detach().item()
        acc_vote = 100*correct_acc_vote / batch_size
        # bayesian cm
        cm = confusion_matrix(y_true=targets.cpu().detach().data, y_pred=predicted_mean.cpu().detach().data, labels=list(range(opt.n_classes)))
    else:
        cm = confusion_matrix(y_true=y.view(-1).cpu().detach().data, y_pred=pred.view(-1).cpu().detach().data, labels=list(range(opt.n_classes)))

    return acc, acc_mean, acc_vote, results[0].detach().item(), cm


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s
