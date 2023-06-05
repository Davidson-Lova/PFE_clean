
# %%
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath("models")), '..'))

# %%
from models import ABCSubSim

# %%
# Here we need to able to extract other flight data from
# the .h5 file, train the model over one flight
# And trying to use the same model for predictions 
# for other flights

import dask.dataframe as dd
import pandas as pd

# %%
import torch
import matplotlib.pyplot as plt
import numpy as np

# %% 
file_name = "../../data/raw/Aircraft_01_dask.h5"
df = dd.read_hdf(file_name, '*')


# %% 
target_name = "N2_1 [% rpm]"
covariable_name = ["N1_1 [% rpm]",
"T1_1 [deg C]",
"ALT [ft]",
"M [Mach]"]


# %%
# Hypperparamètres
N = 5000
lmax = 6
P0 = 0.1
epsilon = 0.01
fact = 0.5
sigma_0 = (lmax + 1) * fact

ns = [4,2,1]

# Temperature initiale
Temp = 10

# Distance qui va être utiliser pour évaluer
# la dissimilarité entre les prédictions y_hat et la réponse y
pdist = 2

args = {"N" : N, "lmax" : lmax, "ns" : ns, "P0" : P0, 
        "epsilon" : epsilon, "fact" : fact, "sigma_0" : sigma_0,
        "Temp" : Temp, "pdist" : pdist}

# %%
model = ABCSubSim.resNet(4,1)


# %%
indexPartition = 0
ty = torch.Tensor(df.partitions[indexPartition][target_name].compute().values).reshape(-1,1)
tyNorm =(ty - ty.mean(axis = 0)) / ty.std(axis = 0)
tX = torch.Tensor(df.partitions[indexPartition][covariable_name].compute().values)
tXNorm = (tX - tX.mean(axis = 0)) / tX.std(axis = 0)

# %%
thetas, rhoMin, epsJ, rhoMax = ABCSubSim.trainBNN(args, tXNorm, tyNorm, model)


# %%
plt.figure()
plt.plot(np.float32(rhoMax), label = 'erreur Max', marker = '^')
plt.plot(np.float32(epsJ), label = 'epsilon_j', marker = '>')
plt.plot(np.float32(rhoMin), label = 'erreur Min', marker = 'v')
plt.yscale('log')
plt.legend()
plt.grid('on')
plt.show()

# %%
y_hats = torch.concatenate(
            tuple([model.set_params(theta).forward(tXNorm) for theta in thetas]), 1)
q2_5 = y_hats.quantile(0.025, 1).reshape(-1,1).detach().numpy()
q97_5 = y_hats.quantile(0.975, 1).reshape(-1,1).detach().numpy()
q25 = y_hats.quantile(0.25, 1).reshape(-1,1).detach().numpy()
q75 = y_hats.quantile(0.75, 1).reshape(-1,1).detach().numpy()
med = y_hats.quantile(0.5, 1).reshape(-1,1).detach().numpy()
plt.clf()
plt.fill_between(np.arange(ty.shape[0]), q2_5[:, 0],
                 q97_5[:, 0], color="#e8e8e8", label='95%')

plt.fill_between(np.arange(ty.shape[0]), q25[:, 0], q75[:, 0],
                 color="#bfbfbf", label='50%')

plt.plot(med, label="mediane", color="#8c0e11")

plt.plot(tyNorm, label='N2_1', color = 'k')

plt.legend()
plt.show()

# %%
indexPartition = 10
ty = torch.Tensor(df.partitions[indexPartition][target_name].compute().values).reshape(-1,1)
tyNorm =(ty - ty.mean(axis = 0)) / ty.std(axis = 0)
tX = torch.Tensor(df.partitions[indexPartition][covariable_name].compute().values)
tXNorm = (tX - tX.mean(axis = 0)) / tX.std(axis = 0)


# %%
model = ABCSubSim.resNet(4,1)

# %%
y_hats = torch.concatenate(
            tuple([model.set_params(theta).forward(tXNorm) for theta in thetas]), 1)
q2_5 = y_hats.quantile(0.025, 1).reshape(-1,1).detach().numpy()
q97_5 = y_hats.quantile(0.975, 1).reshape(-1,1).detach().numpy()
q25 = y_hats.quantile(0.25, 1).reshape(-1,1).detach().numpy()
q75 = y_hats.quantile(0.75, 1).reshape(-1,1).detach().numpy()
med = y_hats.quantile(0.5, 1).reshape(-1,1).detach().numpy()
plt.clf()
plt.fill_between(np.arange(ty.shape[0]), q2_5[:, 0],
                 q97_5[:, 0], color="#e8e8e8", label='95%')

plt.fill_between(np.arange(ty.shape[0]), q25[:, 0], q75[:, 0],
                 color="#bfbfbf", label='50%')

plt.plot(med, label="mediane", color="#8c0e11")

plt.plot(tyNorm, label='N2_1', color = 'k')

plt.legend()
plt.show()

# %%
residuals = y_hats - tyNorm
q2_5 = residuals.quantile(0.025, 1).reshape(-1,1).detach().numpy()
q97_5 = residuals.quantile(0.975, 1).reshape(-1,1).detach().numpy()
q25 = residuals.quantile(0.25, 1).reshape(-1,1).detach().numpy()
q75 = residuals.quantile(0.75, 1).reshape(-1,1).detach().numpy()
med = residuals.quantile(0.5, 1).reshape(-1,1).detach().numpy()
plt.clf()
plt.fill_between(np.arange(ty.shape[0]), q2_5[:, 0],
                 q97_5[:, 0], color="#e8e8e8", label='95%')

plt.fill_between(np.arange(ty.shape[0]), q25[:, 0], q75[:, 0],
                 color="#bfbfbf", label='50%')

plt.hlines(0, xmin = 0, xmax = q2_5.shape[0], label = 'y = 0')

plt.plot(med, label="mediane", color="#8c0e11")

plt.ylim(-3, 3)

plt.legend()
plt.show()
# %%
