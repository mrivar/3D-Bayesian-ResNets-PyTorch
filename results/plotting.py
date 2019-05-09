import matplotlib.pyplot as plt

from plot import *
from plot_utils import *

ds = ["hmdb51", "jhmdb", "ucfsports", "ucf11", "infar"]
sma = 4
dataset = ds[2]
model   = "resnet34"
date    = "04_19"
counter = 1
sets    = "val"

##########################################################
# TRAIN OR VALIDATION ####################################
plot_bayes_vs_freq(dataset, model, date, counter, sets)


## or
csv = [load_csv(complete_name(dataset, model, "03_30", counter, sets)),
       load_csv(complete_name(dataset, "BBB"+model, "03_29", counter, sets))]
plot_against(csv, 'acc', ['Frequentist', 'Bayesian'])
plt.legend()
plt.show()





##########################################################
# TRAIN VS VALID #########################################
plot_total_train_vs_validation(dataset=ds[2], model="BBBresnet34", sma=3)





##########################################################
# BAYESIAN VS FREQUENTIST ################################
plot_total_bayes_vs_freq(dataset=ds[2], model="BBBresnet34", sma=3)






csv = load_csv(complete_name(dataset, "BBB"+model, date, counter, sets))

##########################################################
# UNCERTAINTY ############################################
plot_standard_dev(csv.random_param_mean, csv.random_param_log_alpha)

plot_alpha(csv.random_param_mean, csv.random_param_log_alpha)

plot_weights(csv.random_param_mean, csv.random_param_log_alpha)





##########################################################
# MEAN UNCERTAINTY #######################################
plot_standard_dev(csv.total_param_mean, csv.total_param_log_alpha)

plot_alpha(csv.total_param_mean, csv.total_param_log_alpha)

plot_weights(csv.total_param_mean, csv.total_param_log_alpha)


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
