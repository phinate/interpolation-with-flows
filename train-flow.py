#!/usr/bin/env python
# coding: utf-8

# # Training script for normalizing flows, used for signal MC interpolation

# In[2]:


# boilerplate cell

import torch

from torch import nn
from torch import optim

from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal, StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
)
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.lu import LULinear
from nflows.nn.nets import ResidualNet
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.utils.torchutils import create_mid_split_binary_mask

import os
from glob import glob
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import plothelp
import torch
import random
from functools import partial
from flow_interp.train import train_loop
from sklearn.preprocessing import StandardScaler

def make_scalers(train):
    scaler = StandardScaler()
    mass_scaler = StandardScaler()
    n_features = 2
    scaler.fit(train[:, 0:n_features])
    mass_scaler.fit(train[:, -2:])
    return scaler, mass_scaler

def train_test(data, holdout_points):
    test = list(filter(lambda x: [x[0].m_x, x[0].m_s] in holdout_points, data))
    train = list(filter(lambda x: [x[0].m_x, x[0].m_s] not in holdout_points, data))
    test = np.concatenate(
        [a.to_numpy().tolist() for a in test]
    )  # ['m_bbyy', 'm_jj', 'm_s', 'm_x']
    train = np.concatenate(
        [a.to_numpy().tolist() for a in train]
    )  # ['m_bbyy', 'm_jj', 'm_s', 'm_x']

    return train, test

def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(seed=0)


# In[3]:


# function to specify the type of normalizing flow used.
# this is for a masked autoregressive flow (taken from nflows module)
def flow_maker(num_layers, hidden_features):
    base_dist = ConditionalDiagonalNormal(shape=[2], context_encoder=nn.Linear(2, 4))

    transforms = []
    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=2))
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=2, hidden_features=hidden_features, context_features=2
            )
        )
    transform = CompositeTransform(transforms)

    return Flow(transform, base_dist)


# In[4]:


# import signal MC with pre-selection applied
sig_arr = ak.from_parquet("/eos/user/n/nsimpson/signals-pnn.parquet")

mxms = (
    np.unique(sig_arr[["m_x", "m_s"]])
    .to_numpy()
    .astype([("m_x", int), ("m_s", int)])
    .view((int, 2))
)

# Remove points with m_s =< 15 GeV, since the S is highly boosted and jets aren't resolvable
mxms = mxms[mxms[:, 1] > 15]

sectioned_data = [sig_arr[(sig_arr.m_x == pair[0]) * (sig_arr.m_s == pair[1])] for pair in mxms]


# In[5]:


from sklearn.model_selection import train_test_split

# Set of (mx, ms) points that are *not* trained on
holdout_points = [
    [210,70], 
    [ 245,   90], 
    [ 190,   50],
    [300, 110],
    [500, 170],
    [750, 400]
]

non_test_points = [a for a in mxms.tolist() if a not in holdout_points]
non_test_data = [sig_arr[(sig_arr.m_x == pair[0]) * (sig_arr.m_s == pair[1])] for pair in non_test_points]

train, test = [], []

# For each (mx, ms) point, perform a 70/30 train/valid split
for array in non_test_data:
    tr, te = train_test_split(array[['m_jj', 'm_bbyy_mod', 'm_x', 'm_s', 'total_weight']].to_numpy().tolist(), test_size=0.3, random_state=0, shuffle=True)
    train.append(tr)
    test.append(te)
    
    
# from here, we lose key/value pairs and work with pure numpy.
# the order of entries in the array is:
# ['m_jj', 'm_bbyy', 'm_x', 'm_s', 'total_weight']
train, test = np.concatenate(train), np.concatenate(test)


# In[8]:


# scalers for putting inputs in [0,1]
scalers = reco_scaler, truth_scaler = make_scalers(train[:,:-1])


# In[9]:


# define simple batching mechanism (stolen from JAX docs)

import numpy.random as npr

def batches_w(train, batch_size, rng_state=0):
    num_train = train.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    # batching mechanism
    def data_stream():
        rng = npr.RandomState(rng_state)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                at_batch = train[batch_idx]
                yield at_batch[:,:2], at_batch[:,2:4], at_batch[:,-1]
    return data_stream()

# batch size of 2000
batch_iterator = batches_w(train, 2000)             


# # Training loop

# In[ ]:


for i in range(10):
    print(f"flow {i}")

    # initialize the flow
    flow = flow_maker(
        num_layers=8,
        hidden_features=16,
    )
    
    optimizer = optim.Adam(flow.parameters(), lr=1e-3)
    
    # run the train loop for 10,000 steps
    # "res" is the best flow on the valid set, whose weights we save for later.
    res = train_loop(
        10000,
        flow,
        optimizer,
        batch_iterator=batch_iterator,
        feature_scaler=reco_scaler,
        use_weights=True,
        valid_set=[test[:,:2], test[:,2:4], test[:,-1]],
        context_scaler=truth_scaler,
    )
    torch.save(res, f"flow-all-{i}.pt")
    
# save the scalers to file so we can use them for inference
from joblib import dump

dump(scalers[0], "reco_scaler_large-all.bin", compress=True)
dump(scalers[1], "truth_scaler_large-all.bin", compress=True)


# In[8]:


1


# In[ ]:




