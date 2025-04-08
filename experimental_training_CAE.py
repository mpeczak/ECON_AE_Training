import numpy as np
import pandas as pd
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
from tensorflow import keras
from keras import layers
from keras.layers import Layer
from qkeras import QActivation,QConv2D,QDense,quantized_bits
import qkeras
from qkeras.utils import model_save_quantized_weights
from keras.models import Model
from keras.layers import *
from telescope import *
from utils import *
import inspect
import json

import os
import sys
import graph

import pickle
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import matplotlib.pyplot as plt
import mplhep as hep
from argparse import ArgumentParser

def filter_for_flat_distribution(dataset, index_i):
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
    values_to_balance = np.array(list(dataset.map(lambda features, labels: labels[index_i]).as_numpy_iterator()))
    
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
        
    # Apply filtering to achieve the flat distribution
    filtered_dataset = dataset.enumerate().filter(filter_func).map(lambda idx, data: data)
    
    return filtered_dataset

p = ArgumentParser()
p.add_argument('--mname', type=str)
p.add_argument('--loss', type=str)
p.add_argument('--nepochs', type=int)
p.add_argument('--opath', type=str)
p.add_argument('--mpath', type=str)
p.add_argument('--prepath', type=str)
p.add_argument('--continue_training', action='store_true')
p.add_argument('--batchsize', type=int)
p.add_argument('--lr', type=float)
p.add_argument('--num_files', type=int)
p.add_argument('--optim', type=str)
p.add_argument('--model_per_eLink', action='store_true')
p.add_argument('--model_per_bit_config', action='store_true')
p.add_argument('--biased', type=float)
p.add_argument('--alloc_geom', type=str)
p.add_argument('--low_eta_ft', action='store_true')
p.add_argument('--all_pileup', action='store_true')
p.add_argument('--data_path', type=str)
p.add_argument('--model_per_eta_range', action='store_true')
p.add_argument('--train_scan_parameters', action='store_true')
p.add_argument('--specific_m', type=int)
p.add_argument('--lr_scheduler', type=str)
p.add_argument('--raw_sum_calq', action='store_true')


    
remap_8x8 = [ 4, 12, 20, 28,  5, 13, 21, 29,  6, 14, 22, 30,  7, 15, 23, 31, 
              24, 25, 26, 27, 16, 17, 18, 19,  8,  9, 10, 11,  0,  1,  2,  3, 
              59, 51, 43, 35, 58, 50, 42, 34, 57, 49, 41, 33, 56, 48, 40, 32]

with open('eLink_filts.pkl', 'rb') as f:
    key_df = pickle.load(f)

def apply_cond_transform(wafers, cond):
    # Apply np.exp(cond[:, -2]) - 1 to the last-but-one column of cond
    cond_transformed = tf.concat(
        [cond[:, :-2], tf.expand_dims(tf.exp(cond[:, -2])/200 - 1, axis=-1), cond[:, -1:]],
        axis=-1
    )
    return (wafers, cond_transformed), wafers    
 

    
def get_pams():
    jsonpams={}      
    opt_classes = tuple(opt[1] for opt in inspect.getmembers(tf.keras.optimizers,inspect.isclass))
    for k,v in self.pams.items():
        if type(v)==type(np.array([])):
            jsonpams[k] = v.tolist()
        elif  isinstance(v,opt_classes):
            config = {}
            for hp in v.get_config():
                config[hp] = str(v.get_config()[hp])
            jsonpams[k] = config
        elif  type(v)==type(telescopeMSE8x8):
            jsonpams[k] =str(v) 
        else:
            jsonpams[k] = v 
    return jsonpams

def save_models(autoencoder, name, isQK=False):

    json_string = autoencoder.to_json()
    encoder = autoencoder.get_layer("encoder")
    decoder = autoencoder.get_layer("decoder")
    f'./{model_dir}/{name}.json'
    with open(f'./{model_dir}/{name}.json','w') as f:        f.write(autoencoder.to_json())
    with open(f'./{model_dir}/encoder_{name}.json','w') as f:            f.write(encoder.to_json())
    with open(f'./{model_dir}/decoder_{name}.json','w') as f:            f.write(decoder.to_json())

    autoencoder.save_weights(f'./{model_dir}/{name}.hdf5')
    encoder.save_weights(f'./{model_dir}/encoder_{name}.hdf5')
    decoder.save_weights(f'./{model_dir}/decoder_{name}.hdf5')
    if isQK:
        encoder_qWeight = model_save_quantized_weights(encoder)
        with open(f'{model_dir}/encoder_{name}.pkl','wb') as f:
            pickle.dump(encoder_qWeight,f)
        encoder = graph.set_quantized_weights(encoder,f'{model_dir}/encoder_'+name+'.pkl')
    graph.write_frozen_graph_enc(encoder,'encoder_'+name+'.pb',logdir = model_dir)
    graph.write_frozen_graph_enc(encoder,'encoder_'+name+'.pb.ascii',logdir = model_dir,asText=True)
    graph.write_frozen_graph_dec(decoder,'decoder_'+name+'.pb',logdir = model_dir)
    graph.write_frozen_graph_dec(decoder,'decoder_'+name+'.pb.ascii',logdir = model_dir,asText=True)

    graph.plot_weights(autoencoder,outdir = model_dir)
    graph.plot_weights(encoder,outdir = model_dir)
    graph.plot_weights(decoder,outdir = model_dir)
    


def load_matching_state_dict(model, state_dict_path):
    state_dict = tf.compat.v1.train.load_checkpoint(state_dict_path)
    model_variables = model.trainable_variables
    filtered_state_dict = {}
    for var in model_variables:
        var_name = var.name.split(':')[0]
        if var_name in state_dict:
            filtered_state_dict[var_name] = state_dict[var_name]
    tf.compat.v1.train.init_from_checkpoint(state_dict_path, filtered_state_dict)
    
    
def mean_mse_loss(y_true, y_pred):
    
    max_values = tf.reduce_max(y_true[:,], axis=1)
    
    y_true = tf.gather(K.reshape(y_true,(-1,64)),remap_8x8,axis=-1)
    y_pred = tf.gather(K.reshape(y_pred,(-1,64)),remap_8x8,axis=-1)
    # Calculate the squared difference between predicted and target values
    squared_diff = tf.square(y_pred - y_true)

    # Calculate the MSE per row (reduce_mean along axis=1)
    mse_per_row = tf.reduce_mean(squared_diff, axis=1)
    weighted_mse_per_row = mse_per_row * max_values
    
    # Take the mean of the MSE values to get the overall MSE loss
    mean_mse_loss = tf.reduce_mean(weighted_mse_per_row)
    return mean_mse_loss

def resample_indices(indices, energy, bin_edges, target_count, bin_index):
    bin_indices = indices[(energy > bin_edges[bin_index]) & (energy <= bin_edges[bin_index+1])]
    if len(bin_indices) > target_count:
        return np.random.choice(bin_indices, size=target_count, replace=False)
    else:
        return np.random.choice(bin_indices, size=target_count, replace=True)

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
def custom_resample(wafers,c,simE):
    
    label = (simE[:,0] != 0).astype(int)
    n = len(label)
    print(Counter(label))
    indices = np.expand_dims(np.arange(n),axis = -1)
    # 10x upsample signal
    if args.biased < 0.9:
        over = RandomOverSampler(sampling_strategy=0.1)
        indices_p, label_p = over.fit_resample(indices, label)
    else:
        indices_p, label_p = indices, label
    # downsample until 1:2::pilup:signal
    signal_percent = 1-args.biased
    ratio = args.biased / signal_percent
    if ratio > 1:
        ratio = 1 / ratio
        under = RandomUnderSampler(sampling_strategy=ratio)
        indices_p, label_p = under.fit_resample(indices_p, label_p)
    else:
        under = RandomUnderSampler(sampling_strategy=ratio)
        indices_p, label_p = under.fit_resample(indices_p, label_p)
    print(Counter(label_p))
    wafers_p = wafers[indices_p[:,0]]
    c_p = c[indices_p[:,0]]
    
    return wafers_p, c_p

def custom_reweigh(simE):
    weights = np.log(simE + 1)
    weights /= np.sum(weights)

    return weights

def get_old_mask(eLinks, df):
    # Initialize a mask with all False values, with the same index as the DataFrame
    mask = pd.Series([False] * len(df), index=df.index)

    if eLinks == 5:
        mask = mask | ((df['layer'] <= 11) & (df['layer'] >= 5))
    elif eLinks == 4:
        mask = mask | ((df['layer'] == 7) | (df['layer'] == 11))
    elif eLinks == 3:
        mask = mask | (df['layer'] == 13)
    elif eLinks == 2:
        mask = mask | ((df['layer'] < 7) | (df['layer'] > 13))
    elif eLinks == -1:
        mask = mask | (df['layer'] > 0)
            
    return mask

def load_pre_processed_data(nfiles,batchsize,bits,eta_index = None):
    
    files = os.listdir(os.path.join(args.data_path, f'data_{bits}_eLinks'))
    
    train_files = [f for f in files if 'train' in f][0:nfiles]
    test_files = [f for f in files if 'test' in f][0:nfiles]
    
    # Load and combine all training files
    train_datasets = []
    for file in train_files:
        train_datasets.append(tf.data.experimental.load(os.path.join(args.data_path, f'data_{bits}_eLinks', file)))

    # Combine all loaded training datasets
    train_dataset = train_datasets[0]
    for ds in train_datasets[1:]:
        train_dataset = train_dataset.concatenate(ds)

    # Load and combine all test files
    test_datasets = []
    for file in test_files:
        test_datasets.append(tf.data.experimental.load(os.path.join(args.data_path, f'data_{bits}_eLinks', file)))

    # Combine all loaded test datasets
    test_dataset = test_datasets[0]
    for ds in test_datasets[1:]:
        test_dataset = test_dataset.concatenate(ds)
        
    print("Training dataset size before eta filtering:", train_dataset.cardinality().numpy())
    print("Test dataset size before eta filtering:", test_dataset.cardinality().numpy())


    def filter_by_eta(wafer, cond):
        # Unstack cond to get the eta value; adjust the index if eta is in a different position
        eta, *_ = tf.unstack(cond, axis=-1)

        # Apply the filter condition
        return tf.logical_and(
            tf.greater_equal(eta, eta_min),
            tf.less(eta, eta_max)
        )

    # Step 2: Apply the Filter Using the Named Function
    if eta_index is not None:
        eta_min = eta_bins[eta_index]
        eta_max = eta_bins[eta_index + 1]

        # Use the filter function directly
        train_dataset = train_dataset.filter(filter_by_eta)
        test_dataset = test_dataset.filter(filter_by_eta)
    train_dataset=train_dataset.take(5_000_000)
    test_dataset=test_dataset.take(500_000)
    # Verify the shapes of the final datasets
    print("Training dataset size after eta filter:", train_dataset.cardinality().numpy())
    print("Test dataset size after eta filter:", test_dataset.cardinality().numpy())

    # Prepare the data loaders
    train_loader = train_dataset.batch(batchsize)
    test_loader = test_dataset.batch(batchsize)
    
    # Conditionally map data based on args.raw_sum_calq
    if args.raw_sum_calq:
        # Apply transformation within the map function to modify cond[:, -2] in each batch
        train_loader = train_loader.map(lambda wafers, cond: apply_cond_transform(wafers, cond))
        test_loader = test_loader.map(lambda wafers, cond: apply_cond_transform(wafers, cond))
    else:
        # Only map the data format without transformation
        train_loader = train_loader.map(lambda wafers, cond: ((wafers, cond), wafers))
        test_loader = test_loader.map(lambda wafers, cond: ((wafers, cond), wafers))
    
    # Add this mapping to format the data correctly
#     train_loader = train_loader.map(lambda wafers, cond: ((wafers, cond), wafers))
#     test_loader = test_loader.map(lambda wafers, cond: ((wafers, cond), wafers))

    return train_loader, test_loader
    
def extract_hyperparameters(file_path):
    hyperparams = {}
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            if 'Optimal learning rate' in line:
                hyperparams['learning_rate'] = float(line.split(':')[-1].strip())
            elif 'Optimal batch size' in line:
                hyperparams['batch_size'] = int(line.split(':')[-1].strip())
            elif 'Optimal number of epochs' in line:
                hyperparams['num_epochs'] = int(line.split(':')[-1].strip())
            elif 'Optimal LR scheduler' in line:
                hyperparams['lr_sched'] = line.split(':')[-1].strip()
            elif 'Best performance' in line:
                hyperparams['best_performance'] = float(line.split(':')[-1].strip())
    print('Loaded hyperparameters')
    print(hyperparams)
    return hyperparams

def load_data(nfiles,batchsize,model_info = -1, normalize = True):
    from files import get_rootfiles
    from coffea.nanoevents import NanoEventsFactory
    import awkward as ak
    import numpy as np
    ecr = np.vectorize(encode)
    data_list = []
    simE_list = []
    if args.model_per_eLink:
        eLinks = model_info
    elif args.model_per_bit_config:
        bitsPerOutput = model_info
    
    if args.train_scan_parameters:
        print(os.path.join(args.mpath,f'model_{eLinks}_eLinks'))
        hyper_params = extract_hyperparameters(os.path.join(args.mpath,f'model_{eLinks}_eLinks','hyperparameter_search_results.txt'))
        print(f'Loaded hyperparamters {hyper_params}')
        args.lr = hyper_params['learning_rate']
        args.batchsize = hyper_params['batch_size']
        args.nepochs = hyper_params['num_epochs']
        args.lr_scheduler = hyper_params['lr_sched']
        
        
        

    # Paths to Simon's dataset
    hostid = 'cmseos.fnal.gov'
    basepath = '/store/group/lpcpfnano/srothman/Nov08_2023_ECON_trainingdata'
    tree = 'FloatingpointThreshold0DummyHistomaxDummynTuple/HGCalTriggerNtuple'

    files = get_rootfiles(hostid, basepath)[0:nfiles]

    os.environ["XRD_REQUESTTIMEOUT"] = "120"
    os.environ["XRD_REDIRECTTIMEOUT"] = "120"
    #loop over all the files
    for i,file in enumerate(files):
        x = NanoEventsFactory.from_root(file, treepath=tree).events()
        
        layer = ak.to_pandas(x.wafer.layer)
        eta = ak.to_pandas(x.wafer.eta)
        v = ak.to_pandas(x.wafer.waferv)
        u = ak.to_pandas(x.wafer.waferu)
        wafertype = ak.to_pandas(x.wafer.wafertype)
        wafer_sim_energy = ak.to_pandas(x.wafer.simenergy)
        wafer_energy = ak.to_pandas(x.wafer.energy)
#         gen_pt = ak.to_pandas(x.gen.pt)
        
#         print(len(gen_pt))
#         print(len(eta))
        
        # Combine all DataFrames into a single DataFrame
        data_dict = {
            'eta': eta.values.flatten(),
            'v': v.values.flatten(),
            'u': u.values.flatten(),
            'wafertype': wafertype.values.flatten(),
            'wafer_sim_energy': wafer_sim_energy.values.flatten(),
            'wafer_energy': wafer_energy.values.flatten(),
            'layer': layer.values.flatten(),
#             'gen_pt':gen_pt.values.flatten()
            
        }

        # Add additional features AEin1 to AEin63 to the data dictionary
        key = 'AEin0'
        data_dict[key] = ak.to_pandas(x.wafer[key]).values.flatten()
        for i in range(1, 64):
            key = f'AEin{i}'
            data_dict[key] = ak.to_pandas(x.wafer[key]).values.flatten()
            key = f'CALQ{int(i)}'
            data_dict[key] = ak.to_pandas(x.wafer[key]).values.flatten()
            
        
        # Combine all data into a single DataFrame
        combined_df = pd.DataFrame(data_dict, index=eta.index)
        calq_columns = [f'CALQ{i}' for i in range(1,64)]
        combined_df['sumCALQ'] = combined_df[calq_columns].sum(axis=1)
        
        AEin_columns = [f'AEin{i}' for i in range(1,64)]
        combined_df['AEInSum'] = combined_df[AEin_columns].sum(axis=1)
    
        # Filter by pT
        #         min_pt = -1  # replace with your minimum value
#         gen_pt = ak.to_pandas(x.gen.pt).groupby(level=0).mean()
#         min_pt = 10  # replace with your minimum value
#         max_pt = 10e10  # replace with your maximum value

#         combined_df =combined_df[combined_df['gen_pt'] >= pt_min]
        
        
        
        
        if args.alloc_geom == 'new':
            if args.model_per_eLink:
                filtered_key_df = key_df[key_df['trigLinks'] == float(eLinks)]
                filtered_df = pd.merge(combined_df, filtered_key_df[['u', 'v', 'layer']], on=['u', 'v', 'layer'], how='inner')
            elif args.model_per_bit_config:
                eLinks_with_bit_alloc = [index for index, value in enumerate(bitsPerOutputLink) if value == bitsPerOutput]
                eLinks_with_bit_alloc = [float(b) for b in eLinks_with_bit_alloc if b < 12]
                filtered_key_df = key_df[key_df['trigLinks'].isin(eLinks_with_bit_alloc)]
                filtered_df = pd.merge(combined_df, filtered_key_df[['u', 'v', 'layer']], on=['u', 'v', 'layer'], how='inner')
        
        elif args.alloc_geom =='old':
            if args.model_per_eLink:
                mask = get_old_mask(eLinks, combined_df)
                filtered_df = combined_df[mask]
            elif args.model_per_bit_config:
                eLinks_with_bit_alloc = [index for index, value in enumerate(bitsPerOutputLink) if value == bitsPerOutput]
                eLinks_with_bit_alloc = [b for b in eLinks_with_bit_alloc if b < 6]
                
                mask = get_old_mask(eLinks_with_bit_alloc, combined_df)
#                 print(combined_df)
                filtered_df = combined_df[mask]
                filtered_df = filtered_df.copy()
#                 print(filtered_df)
                
                
        print('Size after eLink filtering')
        print(len(filtered_df))
        
        # Process the filtered DataFrame
        filtered_df.loc[:,'eta'] = filtered_df['eta'] / 3.1
        filtered_df.loc[:,'v'] = filtered_df['v'] / 12
        filtered_df.loc[:,'u'] = filtered_df['u'] / 12
        filtered_df.loc[:,'layer'] = (filtered_df['layer']-1) / 46

        # Convert wafertype to one-hot encoding
        temp = filtered_df['wafertype'].astype(int).to_numpy()
        wafertype_one_hot = np.zeros((temp.size, 3))
        wafertype_one_hot[np.arange(temp.size), temp] = 1

        # Assign the processed columns back to the DataFrame
        filtered_df['wafertype'] = list(wafertype_one_hot)
        filtered_df['sumCALQ'] = np.squeeze(filtered_df['sumCALQ'].to_numpy())
        filtered_df['wafer_sim_energy'] = np.squeeze(filtered_df['wafer_sim_energy'].to_numpy())
        filtered_df['wafer_energy'] = np.squeeze(filtered_df['wafer_energy'].to_numpy())
        filtered_df['layer'] = np.squeeze(filtered_df['layer'].to_numpy())
        

        inputs = []
        for i in range(64):
            cur = filtered_df['AEin%d'%i]
            cur = np.squeeze(cur.to_numpy())
            inputs.append(cur) 
        inputs = np.stack(inputs, axis=-1) #stack all 64 inputs
        inputs = np.reshape(inputs, (-1, 8, 8))
        
        
        layer = filtered_df['layer'].to_numpy()
        eta = filtered_df['eta'].to_numpy()
        v = filtered_df['v'].to_numpy()
        u = filtered_df['u'].to_numpy()
        wafertype = np.array(filtered_df['wafertype'].tolist())
        sumCALQ = filtered_df['sumCALQ'].to_numpy()
        sumCALQ = np.log(sumCALQ+1)
        
        
        wafer_sim_energy = filtered_df['wafer_sim_energy'].to_numpy()
        wafer_energy = filtered_df['wafer_energy'].to_numpy()
        data_list.append([inputs,eta,v,u,wafertype,sumCALQ,layer])
        
        simE_list.append(wafer_sim_energy)
        
         

    
    inputs_list = []
    eta_list = []
    v_list = []
    u_list = []
    wafertype_list = []
    sumCALQ_list = []
    layer_list = []
    
    for item in data_list:
        inputs, eta, v, u, wafertype, sumCALQ,layer = item
        inputs_list.append(inputs)
        eta_list.append(eta)
        v_list.append(v)
        u_list.append(u)
        wafertype_list.append(wafertype)
        sumCALQ_list.append(sumCALQ)
        layer_list.append(layer)

    concatenated_inputs = np.expand_dims(np.concatenate(inputs_list),axis = -1)
    concatenated_eta = np.expand_dims(np.concatenate(eta_list),axis = -1)
    concatenated_v = np.expand_dims(np.concatenate(v_list),axis = -1)
    concatenated_u = np.expand_dims(np.concatenate(u_list),axis = -1)
    concatenated_wafertype = np.concatenate(wafertype_list)
    concatenated_sumCALQ = np.expand_dims(np.concatenate(sumCALQ_list),axis = -1)
    concatenated_layer = np.expand_dims(np.concatenate(layer_list),axis = -1)
    concatenated_simE = np.expand_dims(np.concatenate(simE_list),axis = -1)
    concatenated_cond = np.hstack([concatenated_eta,concatenated_v,concatenated_u, concatenated_wafertype, concatenated_sumCALQ,concatenated_layer])
    
    if args.low_eta_ft:
        print(f'Data before eta fine tuning:{len(concatenated_cond)}')
        mask = (concatenated_eta < 2.1/3.1)[:,0]
        print(mask)
        concatenated_cond = concatenated_cond[mask]
        concatenated_inputs = concatenated_inputs[mask]
        print(f'Data after eta fine tuning:{len(concatenated_cond)}')
        
    events = int(np.min([len(concatenated_cond), 10000000]))
    indices = np.random.permutation(events)
    # Calculate 80% of n
    num_selected = int(0.8 * events)

    # Select the first 80% of the indices
    train_indices = indices[:num_selected]
    test_indices = indices[num_selected:]
    wafer_train = concatenated_inputs[train_indices]
    wafer_test = concatenated_inputs[test_indices]
    
    simE_train = concatenated_simE[train_indices]
    simE_test = concatenated_simE[test_indices]

    cond_train = concatenated_cond[train_indices]
    cond_test = concatenated_cond[test_indices]
    if args.biased:
        wafer_train,cond_train = custom_resample(wafer_train,cond_train,simE_train)
        print(wafer_train.shape)
        wafer_test,cond_test = custom_resample(wafer_test,cond_test, simE_test)
    if args.all_pileup:
        mask_train = (simE_train[:,0] == 0)
        mask_test = (simE_test[:,0] == 0)
        print(len(cond_train))
        wafer_train,cond_train = wafer_train[mask_train],cond_train[mask_train]
        wafer_test,cond_test = wafer_test[mask_test],cond_test[mask_test]
        print(len(cond_train))
    

    weights_train = custom_reweight(simE_train)

    # Create the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((wafer_train,cond_train, weights_train)
    )

    weights_test = custom_reweight(simE_test)

    # Create the test dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((wafer_test,cond_test, weights_test)
    )

    train_loader = train_dataset.batch(batchsize).shuffle(buffer_size=num_selected).prefetch(buffer_size=tf.data.AUTOTUNE)

    test_loader = test_dataset.batch(batchsize).shuffle(buffer_size=events-num_selected).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_loader, test_loader
    
class keras_pad(Layer):
    def call(self, x):
        padding = tf.constant([[0,0],[0, 1], [0, 1], [0, 0]])
        return tf.pad(
        x, padding, mode='CONSTANT', constant_values=0, name=None
    )
    
    
class keras_minimum(Layer):
    def call(self, x, sat_val = 1):
        return tf.minimum(x,sat_val)
    
    
class keras_floor(Layer):
    def call(self, x):
        if isinstance(x, tf.SparseTensor):
            x = tf.sparse.to_dense(x)
            
        return tf.math.floor(x)
      

# Define learning rate schedulers
# def cos_warm_restarts(epoch, total_epochs, initial_lr):
#     """Cosine annealing scheduler with warm restarts."""
#     cos_inner = np.pi * (epoch % (total_epochs // 25))
#     cos_inner /= total_epochs // 25
#     cos_out = np.cos(cos_inner) + 1
#     return float(initial_lr / 2 * cos_out)

def cos_warm_restarts(epoch, total_epochs, initial_lr, min_lr_factor=0.05, restart_period=25):
    """Cosine annealing with warm restarts, with a minimum decay limit."""
    cos_inner = np.pi * (epoch % restart_period) / restart_period
    cos_out = (np.cos(cos_inner) + 1) / 2  # normalize between 0 and 1
    # Scale to ensure it doesn't go below min_lr_factor * initial_lr
    lr = initial_lr * (min_lr_factor + (1 - min_lr_factor) * cos_out)
    return lr

def cosine_annealing(epoch, total_epochs, initial_lr):
    """Cosine annealing scheduler that reduces the learning rate to 1/100 of the initial value."""
    cos_inner = np.pi * (epoch % total_epochs) / total_epochs
    cos_out = np.cos(cos_inner) + 1
    return float((initial_lr / 2) * cos_out * (1 / 25))
# Main function that initializes and trains the model
def train_model(args):
    model_dir = args.opath

    if not os.path.exists(model_dir):
        os.system("mkdir -p " + model_dir)
        
    # Loop through each number of eLinks
    if args.specific_m is not None:
        all_models = [args.specific_m]
    elif args.model_per_eLink:
        if args.alloc_geom == 'old':
            all_models = [2,3,4,5]
        elif args.alloc_geom =='new':
            all_models = [1,2,3,4,5,6,7,8,9,10,11]
    elif args.model_per_bit_config:
        if args.alloc_geom == 'old':
            all_models = [3,5,7,9]
        elif args.alloc_geom =='new':
            all_models = [1,3,5,7,9]



    if args.model_per_eta_range:
        num_eta_bins = 5
        eta_bins =[1.4,1.72,2.04,2.36,2.68,3.0]

    bitsPerOutputLink = [0, 1, 3, 5, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]    


    # Loop through each model configuration (eLinks or bit configurations)
    for m in all_models:
        if args.model_per_eLink:
            eLinks = m
            bitsPerOutput = bitsPerOutputLink[eLinks]
            model_dir = os.path.join(args.opath, f'model_{eLinks}_eLinks')
        elif args.model_per_bit_config:
            bitsPerOutput = m
            model_dir = os.path.join(args.opath, f'model_{bitsPerOutput}_bits')

        if not os.path.exists(model_dir):
            os.system("mkdir -p " + model_dir)

        if args.train_scan_parameters:
            # Load hyperparameters from the search results
            hyper_params = extract_hyperparameters(os.path.join(args.mpath, f'model_{eLinks}_eLinks', 'hyperparameter_search_results.txt'))
            args.lr = hyper_params['learning_rate']
            args.batchsize = hyper_params['batch_size']
            args.nepochs = hyper_params['num_epochs']
            args.lr_scheduler = hyper_params['lr_sched']

        # Define the model architecture
        input_enc = Input(shape=(8, 8, 1), name='Wafer')
        cond = Input(shape=(8,), name='Cond')

        # Encoder
        x = QActivation(quantized_bits(bits=8, integer=1))(input_enc)
        x = keras_pad()(x)
        x = QConv2D(8, 3, strides=2, padding='valid', kernel_quantizer=quantized_bits(bits=6, integer=0))(x)
        x = QActivation(quantized_bits(bits=8, integer=1))(x)
        x = Flatten()(x)
        latent = QDense(16, kernel_quantizer=quantized_bits(bits=6, integer=0))(x)
        latent = QActivation(quantized_bits(bits=9, integer=1))(latent)

        latent = keras_floor()(latent * (1 << 9))
        latent = keras_minimum()(latent / (1 << 9))

        latent = concatenate([latent, cond], axis=1)

        encoder = keras.Model([input_enc, cond], latent, name="encoder")

        # Decoder
        input_dec = Input(shape=(24,))
        y = Dense(24)(input_dec)
        y = ReLU()(y)
        y = Dense(64)(y)
        y = ReLU()(y)
        y = Dense(128)(y)
        y = ReLU()(y)
        y = Reshape((4, 4, 8))(y)
        y = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='valid')(y)
        recon = ReLU()(y[:, 0:8, 0:8])

        decoder = keras.Model(input_dec, recon, name="decoder")

        # Autoencoder
        cae = Model(inputs=[input_enc, cond], outputs=decoder(encoder([input_enc, cond])), name="cae")

        # Define loss and optimizer
        if args.loss == 'mse':
            loss = mean_mse_loss
        elif args.loss == 'tele':
            loss = telescopeMSE8x8
        elif args.loss == 'emd':
            loss = get_emd_loss(args.emd_pth)

        if args.optim == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=args.lr, weight_decay=0.000025)
        elif args.optim == 'lion':
            opt = tf.keras.optimizers.Lion(learning_rate=args.lr, weight_decay=0.00025)

        cae.compile(optimizer=opt, loss=loss)

        # Define learning rate scheduler
        initial_lr = args.lr
        total_epochs = args.nepochs

        if args.lr_scheduler == 'cos_warm_restarts':
            lr_schedule = lambda epoch: cos_warm_restarts(epoch, total_epochs=total_epochs, initial_lr=initial_lr)
        elif args.lr_scheduler == 'cos':
            lr_schedule = lambda epoch: cosine_annealing(epoch, total_epochs=total_epochs, initial_lr=initial_lr)

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

        # Load pre-processed data
        train_loader, test_loader = load_pre_processed_data(args.num_files, args.batchsize, m)

        # Define callbacks
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, 'best-epoch.tf'),
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )

        # Train the model using model.fit
        history = cae.fit(
            train_loader,
            validation_data=test_loader,
            epochs=args.nepochs,
            callbacks=[lr_scheduler, checkpoint_cb]
        )

        # Plot training history
        df = pd.DataFrame(history.history)
        plt.figure(figsize=(10, 6))
        plt.plot(df['loss'], label='Training Loss')
        plt.plot(df['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(model_dir, "loss_plot.png"))
        plt.close()
        
        df.to_csv(f"{model_dir}/df.csv", index=False)

        # Save model weights
        cae.save_weights(os.path.join(model_dir, 'best-epoch.tf'))
        encoder.save_weights(os.path.join(model_dir, 'best-encoder-epoch.tf'))
        decoder.save_weights(os.path.join(model_dir, 'best-decoder-epoch.tf'))

# Usage example
if __name__ == '__main__':
    args = p.parse_args()
    train_model(args)
