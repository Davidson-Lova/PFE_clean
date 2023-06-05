# %%
import dask.dataframe as dd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# %%
file_name = "../../data/raw/Aircraft_01.h5"

store = pd.HDFStore(file_name)
store_keys = store.keys()
with pd.HDFStore(f'{file_name[:-3]}_dask.h5', mode="w") as h :
    for i in range(len(store)) :
        key = store_keys[i]
        h.put(key, store[store_keys[i]], format="table")


# %% 
file_name = "../../data/raw/Aircraft_01_dask.h5"
df = dd.read_hdf(file_name, '*')

# %%
df.describe

# %%
print("column names : ",df.partitions[0].columns)
print("column number : ", len(df.partitions[0].columns))

# %% 
print("number of partitions (i.e. flights) ", df.npartitions)

# %%
nbLines = [df.partitions[i].compute().shape[0] for i in range(df.npartitions)]


# %% 
plt.hist(nbLines, bins=20)

# %% 
target_name = "N2_1 [% rpm]"
covariable_name = ["N1_1 [% rpm]",
"T1_1 [deg C]",
"ALT [ft]",
"M [Mach]"]

# %% 
plt.plot(df.partitions[162][target_name].compute())

# %%
threshold = 5000
indexes =  np.arange(len(nbLines))[np.array(nbLines) <= threshold]

# %%
# for index in indexes :
#     plt.figure()
#     plt.plot(df.partitions[index][target_name].compute().values)
#     plt.savefig('images/N2_{}.png'.format(index), bbox_inches='tight')



# %% 
indexesTrain = random.sample(list(indexes), 10)


# %%
f = open("../../data/processed/trainingSample.txt", "w")
for index in indexesTrain :
    f.write("{}\n".format(index))
f.close()