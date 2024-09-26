import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Concatenate
from qkeras import QActivation, QConv2D, QDense, quantized_bits
import umap
import matplotlib.pyplot as plt

# Cell 2: Define Custom Layers
class keras_pad(tf.keras.layers.Layer):
    def call(self, x):
        padding = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
        return tf.pad(x, padding, mode='CONSTANT', constant_values=0)

    def compute_output_shape(self, input_shape):
        batch_size, height, width, channels = input_shape
        return (batch_size, height + 1, width + 1, channels)

class keras_minimum(tf.keras.layers.Layer):
    def __init__(self, sat_val=1, **kwargs):
        super(keras_minimum, self).__init__(**kwargs)
        self.sat_val = sat_val

    def call(self, x):
        return tf.minimum(x, self.sat_val)

    def compute_output_shape(self, input_shape):
        return input_shape

class keras_floor(tf.keras.layers.Layer):
    def call(self, x):
        return tf.math.floor(x)

    def compute_output_shape(self, input_shape):
        return input_shape

# Cell 3: Build the Encoder Model
# Define input shapes
input_shape = (8, 8, 1)
cond_shape = (8,)

# Inputs
input_enc = Input(shape=input_shape, name='Wafer')
cond = Input(shape=cond_shape, name='Cond')

# Encoder architecture
x = QActivation(activation=quantized_bits(bits=8, integer=1), name='input_quantization')(input_enc)
x = keras_pad()(x)
x = QConv2D(
    filters=8,
    kernel_size=3,
    strides=2,
    padding='valid',
    kernel_quantizer=quantized_bits(bits=6, integer=0, keep_negative=1, alpha=1),
    bias_quantizer=quantized_bits(bits=6, integer=0, keep_negative=1, alpha=1),
    name='conv2d'
)(x)
x = QActivation(activation=quantized_bits(bits=8, integer=1), name='act')(x)
x = Flatten()(x)
x = QDense(
    units=16,
    kernel_quantizer=quantized_bits(bits=6, integer=0, keep_negative=1, alpha=1),
    bias_quantizer=quantized_bits(bits=6, integer=0, keep_negative=1, alpha=1),
    name='dense'
)(x)
x = QActivation(activation=quantized_bits(bits=9, integer=1), name='latent_quantization')(x)
latent = x

# Optional quantization steps
bitsPerOutput = 9
if bitsPerOutput > 0:
    nIntegerBits = 1
    nDecimalBits = bitsPerOutput - nIntegerBits
    outputMaxIntSize = 1 << nDecimalBits
    outputSaturationValue = (1 << nIntegerBits) - 1. / (1 << nDecimalBits)

    latent = keras_floor()(latent * outputMaxIntSize)
    latent = keras_minimum(sat_val=outputSaturationValue)(latent / outputMaxIntSize)

# Concatenate latent vector with condition inputs
latent = Concatenate(axis=1)([latent, cond])

# Build the encoder model
encoder = Model(inputs=[input_enc, cond], outputs=latent, name='encoder')

# Cell 4: Load Weights
encoder.load_weights('encoder_vanilla_AE.hdf5')

import uproot

with uproot.open('ntuple_20.root') as f:
    print(f.keys())  # List top-level keys (e.g., trees)
    if 'FloatingpointThreshold0DummyHistomaxDummynTuple/HGCalTriggerNtuple' in f:
        tree = f['FloatingpointThreshold0DummyHistomaxDummynTuple/HGCalTriggerNtuple']
        print(tree.keys())  # List branches in the tree
    else:
        print("Tree not found in the ROOT file.")



# Cell 5: Prepare Your Data

# Install necessary packages (already installed above)
# Import necessary modules
import numpy as np
import pandas as pd
import tensorflow as tf
import uproot
import awkward as ak
import os
import glob

# Define function to get .root files from the current directory
def get_uploaded_rootfiles():
    rootfiles = glob.glob('*.root')
    return rootfiles

def load_full_data(num_files=-1, eLinks=-1):
    # Get the list of ROOT files in the current directory
    files = get_uploaded_rootfiles()
    if num_files > 0:
        files = files[:num_files]
    print(f"Found {len(files)} ROOT files.")

    # Define the tree name
    tree_name = 'FloatingpointThreshold0DummyHistomaxDummynTuple/HGCalTriggerNtuple'

    # Initialize lists to store data
    inputs_list = []
    eta_list = []
    waferv_list = []
    waferu_list = []
    wafertype_list = []
    sumCALQ_list = []
    layer_list = []

    # Loop over all files
    for i, file in enumerate(files):
        print(f"Processing file {i+1}/{len(files)}: {file}")
        try:
            # Open the file
            with uproot.open(file) as f:
                # Access the tree
                if tree_name in f:
                    tree = f[tree_name]
                else:
                    # Try to find the correct tree
                    keys = f.keys()
                    print(f"Available keys in the file: {keys}")
                    raise ValueError(f"Tree '{tree_name}' not found in file '{file}'.")

                # Define branches to read
                branches = [
                    "gen_pt",
                    "wafer_layer",
                    "wafer_eta",
                    "wafer_waferv",
                    "wafer_waferu",
                    "wafer_wafertype",
                ]
                # Add CALQ and AEin branches
                calq_branches = [f"wafer_CALQ{j}" for j in range(64)]
                aein_branches = [f"wafer_AEin{j}" for j in range(64)]
                branches.extend(calq_branches)
                branches.extend(aein_branches)

                # Read branches
                data = tree.arrays(branches, library="ak")

            # Apply selection criteria
            gen_pt = data["gen_pt"]
            gen_pt_mean = ak.mean(gen_pt, axis=1)
            mask = (gen_pt_mean >= 0) & (gen_pt_mean <= 100000)

            # Apply mask to data
            data = data[mask]

            # Extract variables
            layers = data["wafer_layer"]
            eta = data["wafer_eta"] / 3.1
            waferv = data["wafer_waferv"] / 12
            waferu = data["wafer_waferu"] / 12
            wafertype = data["wafer_wafertype"]

            # Flatten and convert to numpy arrays
            layers_np = ak.to_numpy(ak.flatten(layers))
            eta_np = ak.to_numpy(ak.flatten(eta))
            waferv_np = ak.to_numpy(ak.flatten(waferv))
            waferu_np = ak.to_numpy(ak.flatten(waferu))
            wafertype_np = ak.to_numpy(ak.flatten(wafertype))

            # Assuming wafertype_np contains the wafer types as integers
            N_wafertype = np.unique(wafertype_np).size
            print("Number of unique wafer types:", N_wafertype)

            # One-hot encode wafertype
            temp = wafertype_np.astype(int)
            wafertype_onehot = np.zeros((temp.size, temp.max() + 1))
            wafertype_onehot[np.arange(temp.size), temp] = 1

            # Sum over CALQ
            sumCALQ_np = np.zeros_like(layers_np)
            for j in range(64):
                calq_key = f"wafer_CALQ{j}"
                if calq_key in data.fields:
                    calq_values = data[calq_key]
                    calq_values_flat = ak.to_numpy(ak.flatten(calq_values))
                    sumCALQ_np += calq_values_flat
                else:
                    print(f"{calq_key} not found in the data.")
            sumCALQ_np = np.log(sumCALQ_np + 1)

            # Get inputs
            inputs = []
            for j in range(64):
                aein_key = f"wafer_AEin{j}"
                if aein_key in data.fields:
                    aein_values = data[aein_key]
                    aein_values_flat = ak.to_numpy(ak.flatten(aein_values))
                    inputs.append(aein_values_flat)
                else:
                    print(f"{aein_key} not found in the data.")
            inputs_np = np.stack(inputs, axis=-1)
            inputs_np = np.reshape(inputs_np, (-1, 8, 8))

            # Select eLinks
            select_eLinks = {
                5: (layers_np <= 11) & (layers_np >= 5),
                4: (layers_np == 7) | (layers_np == 11),
                3: (layers_np == 13),
                2: (layers_np < 7) | (layers_np > 13),
                -1: (layers_np > 0)
            }
            selection_mask = select_eLinks[eLinks]
            inputs_np = inputs_np[selection_mask]
            l = (layers_np[selection_mask] - 1) / (47 - 1)
            eta_np = eta_np[selection_mask]
            waferv_np = waferv_np[selection_mask]
            waferu_np = waferu_np[selection_mask]
            wafertype_onehot = wafertype_onehot[selection_mask]
            sumCALQ_np = sumCALQ_np[selection_mask]

            # Append to lists
            inputs_list.append(inputs_np)
            eta_list.append(eta_np)
            waferv_list.append(waferv_np)
            waferu_list.append(waferu_np)
            wafertype_list.append(wafertype_onehot)
            sumCALQ_list.append(sumCALQ_np)
            layer_list.append(l)

        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue

    # Check if data was collected
    if not inputs_list:
        raise ValueError("No data was loaded. Please check your ROOT files and tree names.")

    # Concatenate all data
    concatenated_inputs = np.concatenate(inputs_list)
    concatenated_eta = np.expand_dims(np.concatenate(eta_list), axis=-1)
    concatenated_waferv = np.expand_dims(np.concatenate(waferv_list), axis=-1)
    concatenated_waferu = np.expand_dims(np.concatenate(waferu_list), axis=-1)
    concatenated_wafertype = np.concatenate(wafertype_list)
    concatenated_sumCALQ = np.expand_dims(np.concatenate(sumCALQ_list), axis=-1)
    concatenated_layers = np.expand_dims(np.concatenate(layer_list), axis=-1)

    # Concatenate conditions
    concatenated_cond = np.hstack([
        concatenated_eta,
        concatenated_waferv,
        concatenated_waferu,
        concatenated_wafertype,
        concatenated_sumCALQ,
        concatenated_layers
    ])

    return concatenated_inputs, concatenated_cond

# Specify the number of files to load
num_files = -1  # Set to -1 to load all files, or specify a number

# Load the data
inputs, conditions = load_full_data(num_files)

# Ensure inputs have the correct shape
inputs = np.expand_dims(inputs, axis=-1)  # Add channel dimension if necessary

# Verify the shapes
print('Inputs shape:', inputs.shape)
print('Conditions shape:', conditions.shape)

print("Is eager execution enabled?", tf.executing_eagerly())

tf.config.run_functions_eagerly(True)

# Cell 6: Obtain Latent Representations

# Ensure that the encoder model is defined and weights are loaded (from previous cells)

# Use the encoder to predict the latent representations
batch_size = 32  # Adjust based on your system's memory capacity

latent_representations = encoder.predict([inputs, conditions], batch_size=batch_size)

# Verify the shape of the latent representations
print('Latent representations shape:', latent_representations.shape)

# Cell 7: Apply UMAP and Visualize

import umap
import matplotlib.pyplot as plt

# Initialize UMAP reducer
reducer = umap.UMAP(
    n_neighbors=10,
    min_dist=0.1,
    n_components=3,
    metric='euclidean',
    random_state=42,
    verbose=True,
    n_jobs=-1,
    force_approximation_algorithm=True
)

# Subsample data
num_samples_to_use = 5000
indices = np.random.choice(len(latent_representations), num_samples_to_use, replace=False)
latent_subset = latent_representations[indices]
conditions_subset = conditions[indices]


# Fit and transform the latent representations
embedding = reducer.fit_transform(latent_subset)

# Create a scatter plot


'''
print("conditions_subset vector represents the concatenated conditional inputs used for coloring the UMAP plot.")
print("In this case, the conditions_subset vector contains:")
print(" - Eta")
print(" - Waferv")
print(" - Waferu")
print(" - One-hot encoded Wafertype")
print(" - SumCALQ")
print(" - Layer")

print("\n The first column of conditions_subset (conditions_subset[:, 0]) represents the Eta.")
'''
plt.figure(figsize=(10, 8))
'''
plt.scatter(embedding[:, 0], embedding[:, 1], s=5,  c=conditions_subset[:, 0], cmap='Spectral')
'''
plt.scatter(embedding[:, 0], embedding[:, 1], s=5, cmap='Spectral')
plt.title('UMAP Projection of the Latent Space')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.colorbar(label='Data Points')
plt.show()
plt.savefig("myImagePDF.pdf", format="pdf", bbox_inches="tight")

