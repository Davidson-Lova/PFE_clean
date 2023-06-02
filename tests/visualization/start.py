
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