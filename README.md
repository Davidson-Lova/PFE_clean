# Projet de fin d'étude - BNN - ABC - SS

Support au rapport du PFE

+ `/src/models/` contient l'algorithme même et les réseaux qui seront entrainés
  + `/src/models/cosTube.ipynb` pour le cosinusoïde perturbé
  + `/src/models/sinTube.ipynb` pour le sinusoïde perturbé
  + `/src/models/rotTube.ipynb` pour une application sur des données de vol
  + `/src/models/ABCSubSim.ipynb` code source de l'algo

+ `/src/visualisation` visualisations de l'application de l'algo
  + `/src/visualisation/visCos.ipynb` pour le cosinusoïde perturbé
  + `/src/visualisation/visSin.ipynb` pour le sinusoïde perturbé
  + `/src/visualisation/visRot.ipynb` pour une application sur des données de vol
  + `/src/visualisation/trainingLin.py` 
    On entraine un modèle linéaire sur 7 vols et on test le modèle entrainé sur 3 vols
  + `/src/visualisation/trainingNet.ipynb`
    On entraine un réseau de neurones à une couche cachée sur 7 vols et on test le modèle entrainé sur 3 vols

<!-- Processeur	12th Gen Intel(R) Core(TM) i7-12800H   2.40 GHz

Mémoire RAM installée	32,0 Go (31,7 Go utilisable)

Type du système	Système d’exploitation 64 bits, processeur x64 -->