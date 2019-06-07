import csv
import os


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
        if os.path.isfile(path):
            write_header = False

        self.log_file = open(path, 'a')
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

def calculate_test_accuracy(outputs, targets, opt):
    batch_size = targets.size(0)

    results, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().data.detach().item()
    acc = 100*n_correct_elems / batch_size

    acc_mean, acc_vote = acc, acc

    if opt.bayesian and opt.num_samples > 1:
        # acc_mean
        _, predicted_mean = torch.max(outputs.view(opt.num_samples, batch_size, -1).mean(dim=0).data, 1)
        correct_acc_mean += predicted_mean.cpu().eq(targets.data).sum().detach().item()
        acc_mean = 100*correct_acc_mean / batch_size
        # acc_vote
        votes, _ = pred.view(opt.num_samples, -1).mode(dim=0)
        correct_acc_vote += votes.cpu().eq(targets.data).sum().detach().item()
        acc_vote =100* correct_acc_vote / batch_size

    return acc, acc_mean, acc_vote, results[0].detach().item()

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s
