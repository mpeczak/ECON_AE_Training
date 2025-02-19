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
    ('--opath', p.STR),
    ('--num_files', p.INT),
    ('--model_per_eLink', p.STORE_TRUE),
    ('--model_per_bit_config', p.STORE_TRUE),
    ('--biased', {'type': float}),
    ('--save_every_n_files', p.INT),
    ('--alloc_geom', p.STR)
)
args = p.parse_args()

tree = 'FloatingpointThreshold0DummyHistomaxDummynTuple/HGCalTriggerNtuple'
files = get_rootfiles_local('/home/export/eertorer/scratch/ECONAE/data')[:args.num_files]
simE_list = []
data_list = []

def filter_for_flat_distribution(simE):
    """
    Filters the given TensorFlow dataset to achieve a flat distribution over the specified index i
    of the second element (assumed to be an 8-dimensional tensor) in each dataset element.

    Args:
    - dataset (tf.data.Dataset): The input dataset.
    - index_i (int): The index of the 8-dimensional tensor to achieve a flat distribution over.

    Returns:
    - tf.data.Dataset: A new dataset filtered to achieve a flat distribution across non-zero bins for index_i.
    """
    # Extract the values at index_i from the dataset
    values_to_balance = simE
    
    # Compute histogram over these values
    counts, bins = np.histogram(values_to_balance, bins=10)
    
    # Identify non-zero bins and determine the minimum count across them for a flat distribution
    non_zero_bins = counts > 0
    min_count_in_non_zero_bins = np.min(counts[non_zero_bins])
    
    # Determine which indices to include for a flat distribution
    indices_to_include = []
    current_counts = np.zeros_like(counts)
    for i, value in enumerate(values_to_balance):
        bin_index = np.digitize(value, bins) - 1
        bin_index = min(bin_index, len(current_counts) - 1)  # Ensure bin_index is within bounds
        if current_counts[bin_index] < min_count_in_non_zero_bins:
            indices_to_include.append(i)
            current_counts[bin_index] += 1
            
    # Convert list of indices to a TensorFlow constant for filtering
    indices_to_include_tf = tf.constant(indices_to_include, dtype=tf.int64)
    
    # Filtering function to apply with the dataset's enumerate method
    def filter_func(index, data):
        return tf.reduce_any(tf.equal(indices_to_include_tf, index))
        
    indices_to_include_array = np.array(indices_to_include)  # Ensure indices_to_include is an array
    filtered_simE = concatenated_simE[indices_to_include_array]
    
    return filtered_simE

def custom_resample(simE):
    """
    Upsamples signal (simE != 0) by 10x, then undersamples
    to achieve a ratio of pileup : signal ~ (args.biased) : (1 - args.biased).
    """
    label = (simE[:, 0] != 0).astype(int)  # 1 for signal, 0 for pileup
    n = len(label)

    print("Original label distribution:", Counter(label))

    indices = np.expand_dims(np.arange(n), axis=-1)
    # 10x upsample signal
    over = RandomOverSampler(sampling_strategy=0.1)
    indices_p, label_p = over.fit_resample(indices, label)

    # Downsample until ratio = pileup : signal = args.biased : (1 - args.biased)
    signal_percent = 1 - args.biased
    ratio = args.biased / signal_percent

    if ratio > 1:
        ratio = 1 / ratio
        under = RandomUnderSampler(sampling_strategy=ratio)
        indices_p, label_p = under.fit_resample(indices_p, label_p)
    else:
        under = RandomUnderSampler(sampling_strategy=ratio)
        indices_p, label_p = under.fit_resample(indices_p, label_p)

    print("New label distribution:", Counter(label_p))

    return simE[indices_p[:, 0]]

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
    final = filter_for_flat_distribution(concatenated_simE) #filtered
    #final = custom_resample(concatenated_simE) #resampled
    #final = concatenated_simE #raw input

    

    
plt.figure(figsize=(10, 6))
plt.title("Sim Energy Filtered, Log")
plt.hist(final, bins=100, alpha=0.5, label="Sim Energy", log = True)
plt.xlabel("Sim Energy")
plt.ylabel("Frequency")
plt.legend(loc="upper right")
plt.savefig("/mnt/scratch/mpeczak/ECON_AE_Training/sim_energy_filter_log_y.png", format="png")

plt.show()
