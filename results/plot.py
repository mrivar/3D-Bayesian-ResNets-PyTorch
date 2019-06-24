import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

from plot_utils import *


############################
# iPython
def ipypthon_plot_log(log_files, prefix, suffix='_valid.txt', col='acc', subplot=None, figure=None,
                      labels=None, colors = None, linestyles='-',
                      legend_formatting=None, **kwargs):
    logs_names = [f for f in log_files if f.endswith(suffix) and f.startswith(prefix)]
    logs_names.sort()
    logs = [load_csv(f) for f in logs_names]
    if figure is not None:
        ax = figure.add_subplot(subplot or '111', **kwargs)
        ret = plot_against(logs, col, labels or logs_names, linestyles, colors, ax)
        ax.grid(color="#f1f1f4")
        if legend_formatting: ax.legend(loc=4, framealpha=0)
        else: ax.legend()
    else:
        ret = plot_against(logs, col, labels or logs_names, linestyles, colors)
        plt.grid(color="#f1f1f4")
        if legend_formatting: plt.legend(loc=4, framealpha=0)
        else: plt.legend()
        if subplot is not None: plt.subplot(subplot, **kwargs)
    return ret



def ipypthon_plot_cols(log_files, prefix, suffix='_val.log', cols=['acc', 'acc_mean', 'acc_vote'], labels=['regular', 'averaging', 'voting'], subplot=None, figure=None, **kwargs):
    logs_names = [f for f in log_files if f.endswith(suffix) and f.startswith(prefix) and 'BBB' in f]
    assert len(logs_names) == 1, '%s'%(str(logs_names))
    f = logs_names[0]
    logs = load_csv(f)
    if subplot is None:
        plot_cols_against(logs, cols, labels)
        plt.legend()
    else:
        if figure is None:
            plot_cols_against(logs, cols, labels)
            plt.legend()
            plt.subplot(subplot, **kwargs)
        else:
            ax = figure.add_subplot(subplot, **kwargs)
            plot_cols_against(logs, cols, labels, ax)
            ax.legend()

            
def ipypthon_plot_sd(log_files, prefix, suffix='_uncertainty.log', what_layer="conv", what_sd='random', subplot=None, figure=None, marker_epoch=None, **kwargs):
    logs_names = [f for f in log_files if f.endswith(suffix) and f.startswith(prefix) and 'BBB' in f]
    assert len(logs_names) == 1, '%s'%(str(logs_names))
    assert what_layer in ['conv', 'fc'] and what_sd in ['random', 'total']
    f = logs_names[0]
    log = load_csv(f)
    sds = get_sd_from_alpha(log[what_sd + '_param_log_alpha'], log[what_sd + '_param_mean'])
    if what_layer == "fc": sds = get_sd_from_rho(log[what_sd + '_param_rho'])
    if subplot is None:
        plt.plot(log['epoch'], sds)
        plt.grid(color="#f1f1f4")
    else:
        if figure is None:
            plt.plot(log['epoch'], sds)
            plt.grid(color="#f1f1f4")
            plt.subplot(subplot, **kwargs)
        else:
            ax = figure.add_subplot(subplot, **kwargs)
            plt.plot(log['epoch'], sds)
            if marker_epoch: plt.plot(marker_epoch, sds[marker_epoch], marker="*", markersize=8.5)
            plt.grid(color="#f1f1f4")
            #ax.legend()


def ipypthon_plot_unc(log_files, prefix, suffix='_uncertainty.log', subplot=None, figure=None, what_unc='epistemic', marker_epoch=None, **kwargs):
    logs_names = [f for f in log_files if f.endswith(suffix) and f.startswith(prefix) and 'BBB' in f]
    assert len(logs_names) == 1, '%s'%(str(logs_names))
    assert what_unc in ['epistemic', 'aleatoric']
    f = logs_names[0]
    log = load_csv(f)
    if subplot is None:
        plt.plot(log['epoch'], log[what_unc])
        plt.grid(color="#f1f1f4")
    else:
        if figure is None:
            plt.plot(log['epoch'], log[what_unc])
            plt.grid(color="#f1f1f4")
            plt.subplot(subplot, **kwargs)
        else:
            ax = figure.add_subplot(subplot, **kwargs)
            plt.plot(log['epoch'], log[what_unc])
            if marker_epoch: plt.plot(marker_epoch, log[what_unc][marker_epoch], marker="*", markersize=8.5)
            plt.grid(color="#f1f1f4")
            #ax.legend()



############################
# Plots
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
    plt.ylabel("Accuracy")
    plt.ylim(0,1.0001)
    plt.yticks(np.arange(0.,1.0001,0.1))
    plt.xlabel("Epochs")
    plt.xlim(0, 500)
    plt.grid(color="#f1f1f4")
    plt.legend()
    plt.show()


def plot_total_bayes_vs_freq(dataset, model, sma=1):
    title = "%s_%s vs %s_%s"%(dataset, model, dataset, "BBB"+model)
    f_models = [x for x in os.listdir()
                if os.path.isdir(x) and "%s_%s"%(dataset, model) in x]
    b_models = [x for x in os.listdir()
                if os.path.isdir(x) and "%s_%s"%(dataset, "BBB"+model) in x]
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
    plt.ylabel("Accuracy")
    plt.ylim(0,1.0001)
    plt.yticks(np.arange(0.,1.0001,0.1))
    plt.xlabel("Epochs")
    plt.xlim(0, 500)
    plt.grid(color="#f1f1f4")
    plt.legend()
    plt.show()


def plot_alpha(epochs, log_alpha):
    plot_n_show(epochs, np.exp(log_alpha))


def plot_rho(epochs, rho):
    plot_n_show(epochs, rho)


def plot_std_alpha(mu, log_alpha):
    stds = np.sqrt(pow(mu, 2) * np.exp(log_alpha))
    plt.plot(range(1, len(mu)+1), stds)
    plt.show()

def plot_std_rho(epochs, rho):
    plot_n_show(epochs, np.log1p(np.exp(rho)))

def plot_conv_weights(mu, log_alpha, xlim=0.0002, epochs=[0,19,49,99,499],
    colors=["#A8A7A7", "#CC527A", "#E8175D", "#474747", "#363636"]):
    stds = pow(mu, 2) * np.exp(log_alpha)
    mus = np.asarray(mu[epochs])
    stds  = np.asarray(stds[epochs])

    for i, e in enumerate(epochs):
        # distribution = np.random.normal(loc=mus[i], scale=stds[i], size=500000)
        distribution = np.random.normal(loc=0, scale=stds[i], size=50000)
        sns.distplot(distribution, hist=False, label="%s"%str(e+1), color=colors[i])

    if xlim: plt.xlim(-xlim, xlim)
    plt.show()

def plot_conv_weights_with_mu(mu, log_alpha, epochs=[0,19,49,99,499],
    colors=["#A8A7A7", "#CC527A", "#E8175D", "#474747", "#363636"]):
    stds = pow(mu, 2) * np.exp(log_alpha)
    mus = np.asarray(mu[epochs])
    stds  = np.asarray(stds[epochs])

    for i, e in enumerate(epochs):
        distribution = np.random.normal(loc=mus[i], scale=stds[i], size=500000)
        # distribution = np.random.normal(loc=0, scale=stds[i], size=50000)
        sns.distplot(distribution, hist=False, label="%s"%str(e+1), color=colors[i])

    plt.show()


def plot_fc_weights(mu, rho, xlim=0.0002, epochs=[0,19,49,99,499],
    colors=["#A8A7A7", "#CC527A", "#E8175D", "#474747", "#363636"]):
    stds = np.log1p(np.exp(rho))
    mus = np.asarray(mu[epochs])
    stds  = np.asarray(stds[epochs])

    for i, e in enumerate(epochs):
        # distribution = np.random.normal(loc=mus[i], scale=stds[i], size=500000)
        distribution = np.random.normal(loc=0, scale=stds[i], size=50000)
        sns.distplot(distribution, hist=False, label="%s"%str(e+1), color=colors[i])

    if xlim: plt.xlim(-xlim, xlim)
    plt.show()