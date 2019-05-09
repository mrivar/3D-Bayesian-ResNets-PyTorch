import pandas as pd
import matplotlib.pyplot as plt

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


def plot_against(csv, column, label):
    # @param csv: Array of pandas loaded csv
    # @param column: Column to plot (e.g. 'acc' or 'loss')
    for val, lab in zip(csv, label):
        plt.plot(val.epoch, val[column], label=lab)
