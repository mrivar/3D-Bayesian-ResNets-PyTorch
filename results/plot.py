import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

from plot_utils import *

def plot_train_vs_validation(dataset, model, date, counter):
    sets = ['train', 'val']
    csv = [load_csv(complete_name(dataset, model, date, counter, s))
            for s in sets]
    plot_against(csv, 'acc', sets)


def plot_bayes_vs_freq(dataset, model, date, counter, sets):
    models = ['', 'BBB']
    csv = [load_csv(complete_name(dataset, m+model, date, counter, sets))
            for m in models]
    plot_against(csv, 'acc', ['Frequentist', 'Bayesian'])
    plt.legend()
    plt.show()


def plot_total_train_vs_validation(dataset, model, sma=1):
    title = "%s_%s"%(dataset, model)
    models = [x for x in os.listdir()
                if os.path.isdir(x) and title in x]
    valid = []
    train = []
    for m in models:
        valid.append(np.asarray(load_csv(m+"/val.log").acc))
    for m in models:
        train.append(np.asarray(load_csv(m+"/train.log").acc))
    valid = np.mean(valid, axis=0)
    train = np.mean(train, axis=0)
    if sma > 1:
        valid = apply_sma(valid, sma)
        train = apply_sma(train, sma)
    plt.plot(range(len(train)), train, label="Training")
    plt.plot(range(len(valid)), valid, label="Validation")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_total_bayes_vs_freq(dataset, model, sma=1):
    title = "%s_%s vs %s_%s"%(dataset, model, dataset, "BBB"+model)
    f_models = [x for x in os.listdir()
                if os.path.isdir(x) and "%s_%s"%(ds[dataset], model) in x]
    b_models = [x for x in os.listdir()
                if os.path.isdir(x) and "%s_%s"%(ds[dataset], "BBB"+model) in x]
    freqs = []
    bayes = []
    for m in f_models:
        freqs.append(np.asarray(load_csv(m+"/val.log").acc))
    for m in b_models:
        bayes.append(np.asarray(load_csv(m+"/val.log").acc))
    freqs = np.mean(freqs, axis=0)
    bayes = np.mean(bayes, axis=0)
    if sma > 1:
        freqs = apply_sma(freqs, sma)
        bayes = apply_sma(bayes, sma)
    plt.plot(range(len(freqs)), freqs, label="Frequentist")
    plt.plot(range(len(bayes)), bayes, label="Bayesian")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_alpha(mean, log_alpha):
    alphas = np.exp(log_alpha)
    plt.plot(range(1, len(mean)), alphas)
    plt.show()


def plot_standard_dev(mean, log_alpha):
    stds = pow(mean, 2) * np.exp(log_alpha)
    plt.plot(range(1, len(mean)), stds)
    plt.show()


def plot_weights(mean, log_alpha, epochs=[0,19,49,99,499],
    colors=["#A8A7A7", "#CC527A", "#E8175D", "#474747", "#363636"]):
    stds = pow(mean, 2) * np.exp(log_alpha)
    means = np.asarray(mean[epochs])
    stds  = np.asarray(stds[epochs])

    for i, e in enumerate(epochs):
        #distribution = np.random.normal(loc=means[i], scale=stds[i], size=500000)
        distribution = np.random.normal(loc=0, scale=stds[i], size=50000)
        sns.distplot(distribution, hist=False, label="%s"%str(e+1), color=colors[i])

    plt.xlim(-0.0002, 0.0002)
    plt.show()
