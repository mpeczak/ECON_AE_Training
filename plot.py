import matplotlib.pyplot as plt
import pandas as pd 
import h5py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from telescope import *
from utils import *

p = ArgumentParser()
p.add_args(
    ('--fname', p.STR), #hdf5 info file name
    ('--epath', p.STR), #path to encoder weights
    ('--jpath', p.STR)  #path to encoder json file
)
args = p.parse_args()

with h5py.File(args.fname,'r') as f:
    train_latent = f['train_latent']
    test_latent = f['test_latent'] 
    conds = f['conds']
    inputs_list = f['inputs_list']
    eta_list = f['eta_list']
    waferv_list = f['waferv_list']
    waferu_list = f['waferu_list']
    wafertype_list = f['wafertype_list']
    sumCALQ_list = f['sumCALQ_list']
    layer_list = f['layer_list']
f.close()


with open(args.jpath) as json_file:
    json_config = json_file.read()
encoder = keras.models.model_from_json(json_config)
encoder.load_weights(args.epath)

plt.scatter(encoder[:,0], encoder[:,1], marker='o', s=0.1, c='#d53a26')
plt.show()
