import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

blue = '#1f77b4'
orange = '#ff7f0e'
gray = '#7f7f7f'

def load_csv(name):
    with open(name, 'r') as f:
        return pd.read_csv(f, sep="\t")


def complete_name(dataset, model, date, counter, sets):
    return "%s_%s_%s__%d/%s.log"%(dataset, model, date, counter, sets)


def apply_sma(original, sma=1):
    ders = [0] * (sma + 1)
    for i in range(1 + sma, len(original)):
        derivative = sum(original[i-sma:i]) / sma
        ders.append(derivative)
    return ders


def get_sd_from_alpha(alpha, mu):
    return np.sqrt(np.exp(alpha) * np.power(mu, 2))


def get_sd_from_rho(rho):
    return np.log(1 + np.exp(rho))


def plot_against(csv, column, label, linestyle='-', colors=None, ax=None):
    # @param csv: Array of pandas loaded csv
    # @param column: Column to plot (e.g. 'acc' or 'loss'). Can be a tuple or array. If first value not found, try next.
    if ax is None: ax=plt
    if type(linestyle) == str: linestyle = [linestyle]*len(csv)
    if type(column) == str: column=[column]
    if colors is None: colors = [None]*len(csv)
    for val, lab, ls, cl in zip(csv, label, linestyle, colors):
        for col in column:
            try:
                ax.plot(val.epoch, val[col], linestyle=ls, color=cl, label=lab)
                break
            except:
                continue
    return ax

def plot_cols_against(csv, columns, label, ax=None, linestyle='-'):
    # @param csv: 1 pandas loaded csv
    # @param column: Array of columns to plot (e.g. ['acc', 'acc_mean'] or 'loss')
    if ax is None: ax=plt
    for col, lab in zip(columns, label):
        ax.plot(csv.epoch, csv[col], linestyle, label=lab)


def plot_n_show(y, x):
    plt.plot(y, x)
    plt.show()



def get_max_valid_value_and_train(log_files, bayesian=True, contains=(''), xlim=10000):
    assert type(contains) == list or type(contains) == tuple
    if bayesian:
        valid_model = [f for f in log_files if 'BBB' in f and all(ss in f for ss in contains)][0]
    else:
        valid_model = [f for f in log_files if 'BBB' not in f and all(ss in f for ss in contains)][0]
    valid_model_csv = load_csv(valid_model)['acc_mean'][:xlim]
    max_val = max(valid_model_csv)
    max_val_index = np.argmax(valid_model_csv)
    print("Model: %35s \tMax val: %f in epoch %d"%(valid_model, max_val, max_val_index))

    train_model = valid_model.replace('val', 'train')
    train_model_csv = load_csv(train_model)['acc'][:xlim]
    # max_val = max(train_model_csv)
    # max_val_index = np.argmax(train_model_csv)
    max_val = train_model_csv[max_val_index]
    print("\t\t\t\t\t\tMax val: %f in epoch %d"%(max_val, max_val_index))

    
    
