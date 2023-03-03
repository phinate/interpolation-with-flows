#!/usr/bin/env python
# coding: utf-8

# # Produce plots that assess the results from training the flows

# In[93]:


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


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(seed=0)

# use pNN model that was converted to onnx
onnx_model = '/eos/user/n/nsimpson/onnx_model.onnx'

# import signal MC with preselection
sig_arr = ak.from_parquet("/eos/user/n/nsimpson/signals-pnn.parquet")

# folder with the flows inside
root_path = "/home/jovyan/flow-interpolation-new"

mxms = (
    np.unique(sig_arr[["m_x", "m_s"]])
    .to_numpy()
    .astype([("m_x", int), ("m_s", int)])
    .view((int, 2))
)
# Remove points with m_s =< 15 GeV, since the S is highly boosted and jets aren't resolvable
mxms = mxms[mxms[:, 1] > 15]

# sectioned_data = [sig_arr[(sig_arr.m_x == pair[0]) * (sig_arr.m_s == pair[1])] for pair in mxms]

holdout_points = [
    [210,70], 
    [ 245,   90], 
    [ 190,   50],
    [300, 110],
    [500, 170],
    [750, 400]
]
holdout_points = mxms[37:60]
non_test_points = [a for a in mxms.tolist() if a not in holdout_points]
non_test_data = [sig_arr[(sig_arr.m_x == pair[0]) * (sig_arr.m_s == pair[1])] for pair in non_test_points]
test_data = [sig_arr[(sig_arr.m_x == pair[0]) * (sig_arr.m_s == pair[1])] for pair in holdout_points]


# In[94]:


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

flows = [
    flow_maker(
        num_layers=8,
        hidden_features=16,
    ) for _ in range(10)
]
for flow,i in zip(flows, [0,1,2,3,4,5,6,7,8,9]):
    flow.load_state_dict(torch.load(os.path.join(root_path, f"flow-all-{i}.pt"))[0])


# In[95]:


def ak_to_ndarray(arr):
    return (
        arr.to_numpy()
        .astype([(field, np.float64) for field in arr.fields])
        .view((np.float64, len(arr.fields)))
    )


# In[96]:


from onnx2pytorch import ConvertModel
import onnx

onnx_net = onnx.load(onnx_model)
pytorch_model = ConvertModel(onnx_net,experimental=True)

test_data_hists = []
with torch.no_grad():
    for array in test_data:
        array = ak_to_ndarray(array[['m_jj', 'm_bbyy_mod', 'm_x', 'm_s', 'total_weight']])
        pnn_output = pytorch_model.forward(torch.tensor(array[:,:-1], dtype=torch.float32)).detach().numpy()
        test_data_hists.append(np.histogram(pnn_output.ravel(), weights=array[:,-1]))


# In[97]:


from joblib import load

def load_scalers(path, names=("reco_scaler_large-all.bin", "truth_scaler_large-all.bin")):
    return load(os.path.join(path, names[0])), load(os.path.join(path, names[1]))

feature_scaler, context_scaler = load_scalers(root_path)


# In[98]:


from flow_interp import sample
num_samples = 10000

flow_samples = np.array([sample._sample_flow(flow, feature_scaler, context_scaler, holdout_points, num_samples) for flow in flows])
flow_samples.shape # flows, context, samples, features


# In[99]:


mx, ms  = holdout_points[0]
tmp = np.ones((10000,1))
mx = tmp*mx
ms = tmp*ms
big_arr = np.hstack((mx,ms))

for mx, ms in holdout_points[1:]:
    tmp = np.ones((10000,1))
    mx = tmp*mx
    ms = tmp*ms
    horiz_arr = np.hstack((mx,ms))
    big_arr = np.vstack((big_arr,horiz_arr))
    
big_holdout_points = np.tile(big_arr.reshape(len(holdout_points), 10000, 2), (10, 1, 1, 1))


# In[100]:


flow_data = np.concatenate((flow_samples, big_holdout_points), axis=-1)


# In[101]:


with torch.no_grad():
    pnn_flow = pytorch_model.forward(torch.tensor(flow_data, dtype=torch.float32)).detach().numpy()


# In[102]:


# Number of bins for histogram
bins = 10

# Define the function to apply along the axis
def hist_func(x):
    return np.histogram(x, bins=bins)[0]

test_norms = np.array([np.sum(h[0].ravel()) for h in test_data_hists])
norm_factors = test_norms / 10000

# Apply the function along the third axis (axis=2)
reduced_array = np.apply_along_axis(hist_func, axis=2, arr=np.squeeze(pnn_flow))
fhs = reduced_array*np.tile(norm_factors, (10,1)).reshape(10,len(holdout_points),1)
flow_hists, flow_errs = np.mean(fhs, axis=0), np.std(fhs, axis=0)


# In[103]:


from scipy.stats import chisquare

def plot_hist(dct):
    ax, (data, flow,errs), i = dct["ax"], dct["data"], dct["i"]
#     print(data[0], flow)
    chi2 = chisquare(data[0], f_exp = flow)
    ax.stairs(data[0], data[1], fill=True, label = "signal MC", alpha=0.6)
    ax.stairs(flow, data[1], label="flow avg")
    ax.stairs(flow+errs, data[1], linestyle=":", color='k', label="flow$\pm$$\sigma_{flow}$", alpha=0.7)
    ax.stairs(flow-errs, data[1], linestyle=":", color='k', alpha=0.7)
    mx, ms = holdout_points[i]
    ax.set_title(f"$m_X$, $m_S$ = {mx}, {ms}")#, {chi2[0]/10:.3g}") 
#     ax.set_xlabel("pNN")
    ax.legend(fontsize="6", frameon=False, loc="upper left")

plothelp.autogrid(
    list(zip(test_data_hists, flow_hists, flow_errs)),
    plot_hist,
    subplot_kwargs=dict(facecolor='w', sharex=True),
    title="pNN hists from flow samples (trained on all)"
);


plt.savefig('pnn-interpolation-plots.pdf', bbox_inches="tight")


# In[ ]:





# In[ ]:





# In[ ]:




