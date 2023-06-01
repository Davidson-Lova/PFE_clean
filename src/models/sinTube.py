#%%
import torch
import torch.nn as nn
import numpy as np


# %%
class resLin(nn.Module) :
    def __init__(self, nin, nout) :
        super().__init__()
        self.nin = nin
        self.nout = nout
        self.Layer1 = nn.Linear(nin,nout)
        
    def forward(self, tX) :
        x = self.Layesr1(tX)
        return x
    
    def zero_grad(self, set_to_none: bool = False) -> None:
        return super().zero_grad(set_to_none)
    
    def set_params(self, param_list):
        for p1, p2 in zip(list(self.parameters()), param_list) :
            p1.data = p2
        return self
    

# %%
class resNet(nn.Module) :
    def __init__(self, nin, nout) :
        super().__init__()
        self.nin = nin
        self.nout = nout
        self.Layer1 = nn.Linear(nin, 15)
        self.Layer2 = nn.Linear(15, 15)
        self.Layer3 = nn.Linear(15, nout)

    def forward(self, tX) :
        x = self.Layer1(tX)
        x = torch.relu(x)
        x = self.Layer2(x)
        x = torch.relu(x)
        x = self.Layer3(x)
        return x
    
    def zero_grad(self, set_to_none: bool = False) -> None:
        return super().zero_grad(set_to_none)
    
    def set_params(self, param_list):
        for p1, p2 in zip(list(self.parameters()), param_list) :
            p1.data = p2
        return self
    
    
# %%
def modelSize(ns):
    """ calcul le nombre de poids

    Parameters
    ----------
    ns : list
        liste des nombres de neurones par couches
    Returns
    -------
    int
        Le nombre de poids
    """
    return np.sum([(ns[i]+1)*ns[i+1] for i in range(len(ns)-1)])

# %%
def trainBNN(args, tX, ty, myModel) :
    """ 
    Parameters
    ----------
    args :

    tX :
     
    ty : 

    myModel :

    Returns
    -------
    thetas :

    rhoMin :

    rhoMax : 

    """
    
    nw = modelSize(args["ns"])

    # distribution a priori
    priorDist = torch.distributions.Normal(0, args["sigma_0"])

    modelParamShapes = [p.data.shape for p in list(myModel.parameters())]

    thetas = [([torch.randn(shape) * args["sigma_0"] for shape in modelParamShapes])
          for n in range(args["N"])]

    tYs = torch.concatenate(
        tuple([myModel.set_params(theta).forward(tX) for theta in thetas]), 1)

    # Calcul la dissimilarité
    rhos = (torch.cdist(tYs.t(), ty.t(), p=args["pdist"]) ** 2) / nw

    #
    rhoMin = []
    rhoMax = []
    epsJ = []

    #
    NP0 = int(args["N"]*args["P0"])
    invP0 = int(1/args["P0"])
    j = 0

    # reglage de temperature
    t = 0
    TempCur = args["Temp"]
    while (rhos[0, 0] > args["epsilon"]):

        # On trie les erreurs et on mets les poids dans
        # l'ordre croissant des érreurs qu'ils produisent
        rhos, indices = torch.sort(rhos, 0)

        rhoMin.append(str(np.float32(torch.min(rhos).detach())))
        rhoMax.append(str(np.float32(torch.max(rhos).detach())))

        thetas = [thetas[i] for i in list(indices[:, 0])]

        epsilon_j = rhos[NP0]
        epsJ.append(str(np.float32(epsilon_j[0].detach())))

        # Ici on a un échantillion de taille NP0 et on veut
        # en créer N à partir de cette échantillion en fesant
        # (invPO - 1) pas
        thetasSeeds = thetas[:NP0]
        rhoSeeds = rhos[:NP0]

        # Réglage de sigma_j
        sigma_j = args["sigma_0"] - args["fact"] * args["lmax"]

        #
        thetas = thetasSeeds
        rhos = rhos[:NP0]

        for g in range(invP0 - 1):
            # resampling
            thetasResamples = [[(p + torch.randn(p.shape)*sigma_j)
                                for p in theta] for theta in thetasSeeds]
            # evaluation
            logPriorResamples = [
                [priorDist.log_prob(p) for p in theta] for theta in thetasResamples]
            logPriorSeeds = [
                [priorDist.log_prob(p) for p in theta] for theta in thetasSeeds]

            rj = [[torch.exp(- (p1 - p2)) for (p1, p2) in zip(l1, l2)]
                for (l1, l2) in zip(logPriorResamples, logPriorSeeds)]

            sj = [[(torch.minimum(t, torch.ones(t.shape)))
                for t in theta] for theta in rj]

            bj = [[(torch.rand(t.shape) <= t).float()
                for t in theta] for theta in sj]

            thetasNow = [[(bLoc * tRsLoc + (1 - bLoc) * tSLoc) for (tSLoc, tRsLoc, bLoc)
                        in zip(tS, tRs, b)] for (tS, tRs, b) in zip(thetasSeeds, thetasResamples, bj)]

            tYsNow = torch.concatenate(
                tuple([myModel.set_params(theta).forward(tX) for theta in thetasNow]), 1)
            rhoNow = (torch.cdist(tYsNow.t(), ty.t(), p=args["pdist"]) ** 2) / nw

            thetasVal = [(tNow) if (rhoN <= epsilon_j) else (tSeed)
                        for (tSeed, tNow, rhoN) in zip(thetasSeeds, thetasNow, rhoNow)]
            rhoVal = (rhoNow <= epsilon_j) * (rhoNow - rhoSeeds) + rhoSeeds

            # Mise à jour
            thetasSeeds = thetasVal
            thetas += thetasSeeds
            rhoSeeds = rhoVal
            rhos = torch.concatenate((rhos, rhoSeeds))

            # Réglage de la température
            TempCur = args["Temp"] / np.log(2 + t)
            t += 1

        j += 1
        if (j >= args["lmax"]):
            break

    return thetas, rhoMin, epsJ, rhoMax