
# %%
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import dask.dataframe as dd

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath("models")), '..'))

from models import ABCSubSim
# %%

# %%
# Here we need to able to extract other flight data from
# the .h5 file, train the model over one flight
# And trying to use the same model for predictions
# for other flights


# %%

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

ns = [4, 2, 1]

# Temperature initiale
Temp = 30

# Distance qui va être utiliser pour évaluer
# la dissimilarité entre les prédictions y_hat et la réponse y
pdist = 2

args = {"N": N, "lmax": lmax, "ns": ns, "P0": P0,
        "epsilon": epsilon, "fact": fact, "sigma_0": sigma_0,
        "Temp": Temp, "pdist": pdist}

# %%
model = ABCSubSim.resNet(4, 1)


# %%
f = open('../../data/processed/trainingSample.txt', 'r')
dataIndices = f.read().split('\n')
f.close()

# %%
dataIndices = [int(index) for index in dataIndices]

# %%
trainingIndices = random.sample(dataIndices, 7)
testingIndices = list(set(dataIndices) - set(trainingIndices))


# %%
tXTrain = torch.concatenate(tuple([torch.Tensor(
    df.partitions[index][covariable_name].compute().values) for index in trainingIndices]))

tyTrain = torch.concatenate(tuple([torch.Tensor(df.partitions[index][target_name].compute(
).values).reshape(-1, 1) for index in trainingIndices]))


# %%
tXTest = torch.concatenate(tuple([torch.Tensor(
    df.partitions[index][covariable_name].compute().values) for index in testingIndices]))

tyTest = torch.concatenate(tuple([torch.Tensor(df.partitions[index][target_name].compute(
).values).reshape(-1, 1) for index in testingIndices]))


# %%
tX = torch.concatenate((tXTrain, tXTest))
ty = torch.concatenate((tyTrain, tyTest))


# Normalisation
# %%
tXTrainNorm = (tXTrain - tX.mean(axis=0)) / tX.std(axis=0)

tyTrainNorm = (tyTrain - ty.mean(axis=0)) / ty.std(axis=0)

# %%
tXTestNorm = (tXTest - tX.mean(axis=0)) / tX.std(axis=0)

tyTestNorm = (tyTest - ty.mean(axis=0)) / ty.std(axis=0)


# %%
# Training the BNN
thetas, rhoMin, epsJ, rhoMax = ABCSubSim.trainBNN(args, tXTrainNorm, tyTrainNorm, model)

# %%
# BaseLine of linear regression
aHat = torch.linalg.solve(tXTrainNorm.t() @ tXTrainNorm,
                      tXTrainNorm.t() @ tyTrainNorm)
yHat = np.matmul(tXTrainNorm, aHat)

# %%
# Error plots
rhoFloor = torch.ones(lmax) *  (torch.cdist(tyTrainNorm.t(), yHat.t(), p=pdist) ** 2) / yHat.shape[0]
plt.figure()
plt.plot(np.float32(rhoMax), label = 'erreur Max', marker = '^')
plt.plot(np.float32(epsJ), label = 'epsilon_j', marker = '>')
plt.plot(np.float32(rhoMin), label = 'erreur Min', marker = 'v')
plt.plot(rhoFloor.t(), label = 'erreur RegLin', marker = 'd')
plt.yscale('log')
plt.legend()
plt.grid('on')
plt.show()

# Evaluation of the error over the training set
# %%
rhoTrain = []
for index in trainingIndices : 
    tXloc = torch.Tensor(df.partitions[index][covariable_name].compute().values)
    tXloc = (tXloc - tX.mean(axis = 0)) / tX.std(axis = 0)

    tyLoc = torch.Tensor(df.partitions[index][target_name].compute().values).reshape(-1, 1)
    tyLoc = (tyLoc - ty.mean(axis = 0)) / ty.std(axis = 0)
    tYsLoc = torch.concatenate(
                tuple([model.set_params(theta).forward(tXloc) for theta in thetas]), 1)
    rhoLoc = (torch.cdist(tYsLoc.t(), tyLoc.t(), p=args["pdist"]) ** 2) / tyLoc.shape[0]
    rhoTrain.append(rhoLoc.detach())


# Evaluation of the error over the testing set
# %%
rhoTest = []
for index in testingIndices : 
    tXloc = torch.Tensor(df.partitions[index][covariable_name].compute().values)
    tXloc = (tXloc - tX.mean(axis = 0)) / tX.std(axis = 0)

    tyLoc = torch.Tensor(df.partitions[index][target_name].compute().values).reshape(-1, 1)
    tyLoc = (tyLoc - ty.mean(axis = 0)) / ty.std(axis = 0)
    tYsLoc = torch.concatenate(
                tuple([model.set_params(theta).forward(tXloc) for theta in thetas]), 1)
    rhoLoc = (torch.cdist(tYsLoc.t(), tyLoc.t(), p=args["pdist"]) ** 2) / tyLoc.shape[0]
    rhoTest.append(rhoLoc.detach())


# %%
plt.boxplot([ _.ravel() for _ in rhoTrain])
plt.ylim(0, 0.4)
plt.show()

plt.boxplot([ _.ravel() for _ in rhoTest])
plt.ylim(0, 0.4)
plt.show()

# %%
# little test
for indexPartition in trainingIndices :
    ty = torch.Tensor(df.partitions[indexPartition][target_name].compute().values).reshape(-1,1)
    tyNorm =(ty - ty.mean(axis = 0)) / ty.std(axis = 0)
    tX = torch.Tensor(df.partitions[indexPartition][covariable_name].compute().values)
    tXNorm = (tX - tX.mean(axis = 0)) / tX.std(axis = 0)

    y_hats = torch.concatenate(
                tuple([model.set_params(theta).forward(tXNorm) for theta in thetas]), 1)
    q2_5 = y_hats.quantile(0.025, 1).reshape(-1,1).detach().numpy()
    q97_5 = y_hats.quantile(0.975, 1).reshape(-1,1).detach().numpy()
    q25 = y_hats.quantile(0.25, 1).reshape(-1,1).detach().numpy()
    q75 = y_hats.quantile(0.75, 1).reshape(-1,1).detach().numpy()
    med = y_hats.quantile(0.5, 1).reshape(-1,1).detach().numpy()
    
    plt.fill_between(np.arange(ty.shape[0]), q2_5[:, 0],
                    q97_5[:, 0], color="#e8e8e8", label='95%')

    plt.fill_between(np.arange(ty.shape[0]), q25[:, 0], q75[:, 0],
                    color="#bfbfbf", label='50%')

    plt.plot(med, label="mediane", color="#8c0e11")

    plt.plot(tyNorm, label='N2_1', color = 'k')
    plt.ylim(-4,1)
    plt.legend()
    plt.show()


# %%
for indexPartition in testingIndices :
    ty = torch.Tensor(df.partitions[indexPartition][target_name].compute().values).reshape(-1,1)
    tyNorm =(ty - ty.mean(axis = 0)) / ty.std(axis = 0)
    tX = torch.Tensor(df.partitions[indexPartition][covariable_name].compute().values)
    tXNorm = (tX - tX.mean(axis = 0)) / tX.std(axis = 0)

    y_hats = torch.concatenate(
                tuple([model.set_params(theta).forward(tXNorm) for theta in thetas]), 1)
    q2_5 = y_hats.quantile(0.025, 1).reshape(-1,1).detach().numpy()
    q97_5 = y_hats.quantile(0.975, 1).reshape(-1,1).detach().numpy()
    q25 = y_hats.quantile(0.25, 1).reshape(-1,1).detach().numpy()
    q75 = y_hats.quantile(0.75, 1).reshape(-1,1).detach().numpy()
    med = y_hats.quantile(0.5, 1).reshape(-1,1).detach().numpy()

    plt.fill_between(np.arange(ty.shape[0]), q2_5[:, 0],
                    q97_5[:, 0], color="#e8e8e8", label='95%')

    plt.fill_between(np.arange(ty.shape[0]), q25[:, 0], q75[:, 0],
                    color="#bfbfbf", label='50%')

    plt.plot(med, label="mediane", color="#8c0e11")

    plt.plot(tyNorm, label='N2_1', color = 'k')
    plt.ylim(-4,1)
    plt.legend()
    plt.show()


# %%
