import pandas as pd
import matplotlib.pyplot as plt

ds = ["hmdb51", "jhmdb", "ucfsports", "ucf11"]
sma = 20
dataset = 1

##########################################################
# TRAIN ##################################################
# LOAD TRAIN FILES
model   = "resnet34"
date    = "03_30__1"
sets    = "train"

path = "%s_%s_%s/%s.log"%(ds[dataset], model, date, sets)
with open(path, 'r') as f:
    frequentist = pd.read_csv(f, sep="\t")


dataset = 1
model   = "BBBresnet34"
date    = "03_29__1"
sets    = "train"

path = "%s_%s_%s/%s.log"%(ds[dataset], model, date, sets)
with open(path, 'r') as f:
    bayesian = pd.read_csv(f, sep="\t")



# PLOTTING
plt.plot(frequentist.epoch, frequentist.acc)
plt.plot(bayesian.epoch, bayesian.acc)
plt.show()






##########################################################
# VALIDATION #############################################
sets = "val"
sma = 4

model   = "resnet34"
date    = "03_30__1"
path = "%s_%s_%s/%s.log"%(ds[dataset], model, date, sets)
with open(path, 'r') as f:
    frequentist_val = pd.read_csv(f, sep="\t")

model   = "BBBresnet34"
date    = "03_29__1"
path = "%s_%s_%s/%s.log"%(ds[dataset], model, date, sets)
with open(path, 'r') as f:
    bayesian_val = pd.read_csv(f, sep="\t")


# SMOOTHING USING SIMPLE MOVING AVERAGE
frq_ders = [0] * (sma + 1)
for i in range(1 + sma, len(frequentist_val)):
    derivative = sum(frequentist_val.acc[i-sma:i]) / sma
    frq_ders.append(derivative)

bay_ders = [0] * (sma + 1)
for i in range(1 + sma, len(bayesian_val)):
    derivative = sum(bayesian_val.acc[i-sma:i]) / sma
    bay_ders.append(derivative)

# PLOTTING
plt.plot(frequentist_val.epoch, frq_ders)
plt.plot(bayesian_val.epoch, bay_ders)
plt.show()


# PLOTTING WITHOUT SMOOTHING
plt.plot(frequentist_val.epoch, frequentist_val.acc)
plt.plot(bayesian_val.epoch, bayesian_val.acc)
plt.show()






##########################################################
# UNCERTAINTY ############################################
plt.plot(bayesian_val.epoch, bayesian_val.random_param_logvar)
plt.show()

import seaborn as sns
import numpy as np
epochs= [0,19,49,99,499]
colors= ["#bfd6f6", "#8dbdff", "#64a1f4", "#4a91f2", "#3b7dd8"]
means = bayesian_val.random_param_mean[epochs]
stds  = bayesian_val.random_param_logvar[epochs]

for i, e in enumerate(epochs):
    distribution = np.random.normal(loc=means[i], scale=stds[i], size=500000)
    sns.distplot(distribution, hist=False, label="%s"%str(e+1), color=colors[i])

"""
x = np.random.normal(loc=0, scale=3, size=500000)
y = np.random.normal(loc=0, scale=3.5, size=500000)
z = np.random.normal(loc=0, scale=4, size=500000)
w = np.random.normal(loc=0, scale=4.5, size=500000)
u = np.random.normal(loc=0, scale=5, size=500000)
sns.distplot(x, hist=False, label="1", color="#3b7dd8")
sns.distplot(y, hist=False, label="2", color="#4a91f2")
sns.distplot(z, hist=False, label="3", color="#64a1f4")
sns.distplot(w, hist=False, label="4", color="#8dbdff")
sns.distplot(u, hist=False, label="5", color="#bfd6f6")
plt.xlim([15,-15])
plt.show()
"""



##########################################################
# MEAN UNCERTAINTY #######################################
plt.plot(bayesian_val.epoch, bayesian_val.total_param_logvar)
plt.show()

import seaborn as sns
import numpy as np
epochs= [0,19,49,99,499]
colors= ["#bfd6f6", "#8dbdff", "#64a1f4", "#4a91f2", "#3b7dd8"]
means = bayesian_val.total_param_mean[epochs]
stds  = bayesian_val.total_param_logvar[epochs]

for i, e in enumerate(epochs):
    distribution = np.random.normal(loc=means[i], scale=stds[i], size=500000)
    sns.distplot(distribution, hist=False, label="%s"%str(e+1), color=colors[i])
