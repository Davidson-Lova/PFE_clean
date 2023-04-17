import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time


def segment(theta, ns):
    """ Segment un vecteur theta
    Cette fonction sert à segmenter un theta en une 
    liste de matrices qui est spécifié par ns qui 
    donne le nombre de neurones par couche

    Parameters
    ----------
    theta : torch.Tensor
        Le concatenation des paramètres du réseaux
    ns : list 
        La liste des nombres de neurones pour chaque
        couche du réseaux

    Returns
    -------
    list
        La liste des tenseurs à chaque couche du réseau
    """

    total_len = np.sum([(ns[i]+1)*ns[i+1] for i in range(len(ns)-1)])
    if (total_len != len(theta)):
        print("Error : Wrong dimensions\n")
        return

    param_list = []
    index = 0
    for i in range(len(ns)-1):

        W = torch.Tensor(
            theta[index: (index + ns[i]*ns[i + 1])]).reshape(ns[i+1], ns[i])
        index += ns[i]*ns[i + 1]

        b = torch.Tensor(theta[index: (index + ns[i+1])]).reshape(ns[i+1])
        index += ns[i + 1]

        param_list.append(W)
        param_list.append(b)

    return param_list


def wrap(param_list):
    """ Concatene les paramètres
    Cette fonction sert à concatener les tenseurs
    à chaque couche de réseau en un gros vecteur

    Parameters
    ----------
    param_list : list
        liste qui contient les tenseurs

    Returns
    -------
    torch.Tensor
        Le gros tenseur
    """
    listLen = len(param_list)

    theta = param_list[0].reshape(np.prod(param_list[0].size()), 1)

    for i in range(1, listLen):
        theta = torch.concat(
            (theta, param_list[i].reshape(np.prod(param_list[i].size()), 1)))

    return theta


class FNN(nn.Module):
    """ Definition d'un FNN
    """

    def __init__(self, ns, activation, theta=None):
        """ Initialisation
        Construction d'une instance de la class FNN 
        à partir des paramètres données

        Parameters
        ----------
        ns : list 
            une liste de nombres de couches
        theta : torch.Tensor (None)
            une tenseur de paramètres

        Returns
        -------
        FNN
            L'instance de la classe
        """
        super(FNN, self).__init__()
        self.funclist = []
        self.nbLayers = len(ns)
        self.ns = ns
        self.activation = activation
        for i in range(self.nbLayers-1):
            self.funclist.append(nn.Linear(ns[i], ns[i+1]))
        if (theta != None):
            self.update_weights(theta)

    def forward(self, x):
        """ Evaluation
        Cette fonction fait une forward pass du FNN

        Parameters
        ----------
        x : torch.Tensor
            la valeur d'entré
        Returns
        -------
        torch.Tensor
            Le résultat du forward pass
        """
        for i in range(self.nbLayers - 2):
            x = self.funclist[i](x)
            if (self.activation == 'sigmoid'):
                x = torch.sigmoid(x)
            if (self.activation == 'tanh'):
                x = torch.tanh(x)
            if (self.activation == 'relu'):
                x = torch.relu(x)
        x = self.funclist[self.nbLayers - 2](x)
        return x

    def update_weights(self, thetas):
        """ Mets à jour les paramètres (poids)
        Remplace les poids courants

        Parameters
        ----------
        thetas : torch.Tensor
                liste qui contient les tenseurs

        Returns
        -------
        """
        param_list = segment(thetas, self.ns)
        with torch.no_grad():
            for i in range(self.nbLayers - 1):
                self.funclist[i].weight = nn.Parameter(param_list[i*2])
                self.funclist[i].bias = nn.Parameter(param_list[i*2 + 1])

        return self

    def getTheta(self):
        """ renvoie les poids

        Parameters
        ----------

        Returns
        -------
        torch.Tensor
            Le gros tenseur de poids
        """
        param_list = []
        for i in range(self.nbLayers - 1):
            param_list.append(self.funclist[i].weight)
            param_list.append(self.funclist[i].bias)

        return wrap(param_list)


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

def plotTubeMedian(XT, y, thetas, ns, activation):
    """ plot un tube centré sur la médiane
    Fait un forward pass avec en entrée XT pour chaque 
    theta dans theta, on evalue ensuite la mediane
    le 1er et le 3ème quartile, les quantiles 2.5% et 97.5%
    En on plot le fil de la médiane, et le tube 1er et 3ème
    quartile et le tube quantile 2.5% et 99.75%
    Parameters
    ----------
    XT : torch.Tensor
        tenseur d'entrée
    y : torch.Tensor 
        Solution bruité
    thetas : list of torch.Tensor
        un échantillion de poids
    ns : list 
        Le nombre de neurones par couche
    activation : string
        Fonction d'activation
    Returns
    -------
    """

    ymax = torch.max(y)
    ymin = torch.min(y)
    yspan = ymax - ymin

    myModel = FNN(ns, activation)

    N = thetas.shape[1]
    y_hats = torch.concat(
        tuple([myModel.update_weights(thetas[:, i]).forward(XT) for i in range(0, N)]), 1)

    q2_5 = y_hats.quantile(0.025, 1).reshape(y.shape).detach().numpy()
    q97_5 = y_hats.quantile(0.975, 1).reshape(y.shape).detach().numpy()
    q25 = y_hats.quantile(0.25, 1).reshape(y.shape).detach().numpy()
    q75 = y_hats.quantile(0.75, 1).reshape(y.shape).detach().numpy()

    plt.fill_between(XT.ravel(), q2_5[:, 0],
                     q97_5[:, 0], color="#e8e8e8",  label='95%')
    plt.fill_between(XT.ravel(), q25[:, 0],
                     q75[:, 0], color="#bfbfbf", label='50%')
    plt.plot(XT, y_hats.quantile(0.5, 1).reshape(
        y.shape).detach().numpy(), label="Médiane", color="#8c0e11")

    plt.scatter(XT, y, marker='+', color='k', label='Données d\'entrainement')
    plt.ylim(ymin - 0.01*yspan, ymax + 0.01*yspan)
    plt.legend()


def plotTubeMedianBig(XT, XTBig, y, thetas, ns, activation, c):
    """ même chose que celui d'avant mais sur un domaine plus grands
    Parameters
    ----------
    XT : torch.Tensor
        tenseur d'entrée
    XTBig : torch.Tensor
        tenseur d'entrée grand
    y : torch.Tensor 
        Solution bruité
    thetas : list of torch.Tensor
        un échantillion de poids
    ns : list 
        Le nombre de neurones par couche
    activation : string
        La fonction d'activation
    c : float
        Zoom
    Returns
    -------
    """
    ymax = torch.max(y)
    ymin = torch.min(y)
    yspan = ymax - ymin

    myModel = FNN(ns, activation)

    N = thetas.shape[1]
    y_hats = torch.concat(
        tuple([myModel.update_weights(thetas[:, i]).forward(XTBig) for i in range(0, N)]), 1)

    q2_5 = y_hats.quantile(0.025, 1).reshape(
        (XTBig.shape[0], 1)).detach().numpy()
    q97_5 = y_hats.quantile(0.975, 1).reshape(
        (XTBig.shape[0], 1)).detach().numpy()
    q25 = y_hats.quantile(0.25, 1).reshape(
        (XTBig.shape[0], 1)).detach().numpy()
    q75 = y_hats.quantile(0.75, 1).reshape(
        (XTBig.shape[0], 1)).detach().numpy()
    mediane = y_hats.quantile(0.5, 1).reshape(
        XTBig.shape).detach().numpy()
    plt.fill_between(XTBig.ravel(), q2_5[:, 0],
                     q97_5[:, 0], color="#e8e8e8", label='95%')

    plt.fill_between(XTBig.ravel(), q25[:, 0], q75[:, 0],
                     color="#bfbfbf", label='50%')

    plt.plot(XTBig, mediane, label="Médiane", color='#8c0e11')

    plt.scatter(XT, y, marker='+', color='k', label='Données')
    plt.ylim(ymin - c*yspan, ymax + c*yspan)
    plt.legend()
