import pandas as pd
import matplotlib.pyplot as plt

ds = ["hmdb51", "jhmdb", "ucfsports", "ucf11"]
sma = 20
dataset = 1

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
