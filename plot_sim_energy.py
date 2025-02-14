import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from coffea.nanoevents import NanoEventsFactory
import awkward as ak
from utils.files import get_rootfiles_local, get_files_recursive_local
from utils.utils import ArgumentParser, encode

p = ArgumentParser()
p.add_args(
    ('--num_files', p.INT),
)
args = p.parse_args()

tree = 'FloatingpointThreshold0DummyHistomaxDummynTuple/HGCalTriggerNtuple'
files = get_rootfiles_local('/home/export/eertorer/scratch/ECONAE/data')[:args.num_files]
simE_list = []

for file in enumerate(files):
    print(file[1])
    events = NanoEventsFactory.from_root(file[1], treepath=tree).events()

    min_pt, max_pt = -1, 1e15
    gen_pt = ak.to_pandas(events.gen.pt).groupby(level=0).mean()
    mask = (gen_pt['values'] >= min_pt) & (gen_pt['values'] <= max_pt)

    wafer_sim_energy_pd = ak.to_pandas(events.wafer.simenergy)
    wafer_sim_energy_arr = np.array(wafer_sim_energy_pd.values.flatten())
    simE_list.append(wafer_sim_energy_arr)
    concatenated_simE = np.expand_dims(np.concatenate(simE_list), axis=-1)
    #concatenated_simE = concatenated_simE[concatenated_simE > 0]

    
plt.figure(figsize=(10, 6))
plt.title("Sim Energy > 0, Log Plot")
plt.hist(concatenated_simE, bins=100, alpha=0.5, label="Sim Energy")
plt.xlabel("Sim Energy")
plt.ylabel("Frequency")
plt.legend(loc="upper right")
plt.savefig("/mnt/scratch/mpeczak/ECON_AE_Training/sim_energy.png", format="png")

plt.show()
