#!/usr/bin/env python
# # Produce plots that assess the results from training the flows
# In[93]:
from __future__ import annotations

import json
import os
import random

import awkward as ak
import numpy as np
import onnx
import torch
import uproot
from flow_interp import sample
from joblib import load
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.flows.base import Flow
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
)
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import ReversePermutation
from onnx2pytorch import ConvertModel
from sklearn.preprocessing import StandardScaler
from torch import nn


def make_scalers(train):
    scaler = StandardScaler()
    mass_scaler = StandardScaler()
    n_features = 2
    scaler.fit(train[:, 0:n_features])
    mass_scaler.fit(train[:, -2:])
    return scaler, mass_scaler


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(seed=0)

# Load config details from JSON file
with open('config.json') as f:
    config = json.load(f)

onnx_model = config['onnx_model']
sig_arr_path = config['sig_arr_path']
root_path = config['path_to_flows']
points_to_infer = config["points_to_infer"]
# points_to_infer = [[a["mx"],a["ms"]] for a in infer_dict]
# test_norms = [a["sum_of_total_weights"] for a in infer_dict]
# bins = 
num_layers = config['num_layers']
hidden_features = config['hidden_features']
feature_scaler_path = config['feature_scaler_path']
context_scaler_path = config['context_scaler_path']
num_samples = config['num_samples']
tree_dir = config['samples_filename']

# Import signal MC with preselection
sig_arr = ak.from_parquet(sig_arr_path)

# Load pNN model converted to onnx
onnx_net = onnx.load(onnx_model)


mxms = (
    np.unique(sig_arr[['m_x', 'm_s']])
    .to_numpy()
    .astype([('m_x', int), ('m_s', int)])
    .view((int, 2))
)
# Remove points with m_s =< 15 GeV, since the S is highly boosted and jets aren't resolvable
mxms = mxms[mxms[:, 1] > 15]

# sectioned_data = [sig_arr[(sig_arr.m_x == pair[0]) * (sig_arr.m_s == pair[1])] for pair in mxms]


non_test_points = [a for a in mxms.tolist() if a not in points_to_infer]
non_test_data = [
    sig_arr[(sig_arr.m_x == pair[0]) *
            (sig_arr.m_s == pair[1])] for pair in non_test_points
]
test_data = [
    sig_arr[(sig_arr.m_x == pair[0]) * (sig_arr.m_s == pair[1])]
    for pair in points_to_infer
]


# In[94]:


def flow_maker(num_layers, hidden_features):
    base_dist = ConditionalDiagonalNormal(
        shape=[2], context_encoder=nn.Linear(2, 4),
    )

    transforms = []
    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=2))
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=2, hidden_features=hidden_features, context_features=2,
            ),
        )
    transform = CompositeTransform(transforms)

    return Flow(transform, base_dist)


flows = [
    flow_maker(
        num_layers=8,
        hidden_features=16,
    ) for _ in range(10)
]
for flow, i in zip(flows, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    flow.load_state_dict(
        torch.load(
            os.path.join(root_path, f'flow-all-{i}.pt'),
        )[0],
    )


# In[95]:


def ak_to_ndarray(arr):
    return (
        arr.to_numpy()
        .astype([(field, np.float64) for field in arr.fields])
        .view((np.float64, len(arr.fields)))
    )


# In[96]:


onnx_net = onnx.load(onnx_model)
pytorch_model = ConvertModel(onnx_net, experimental=True)




feature_scaler, context_scaler = load(
    feature_scaler_path,
), load(context_scaler_path)


flow_samples = np.array([
    sample._sample_flow(
        flow, feature_scaler, context_scaler, points_to_infer, num_samples,
    ) for flow in flows
])
flow_samples.shape  # flows, context, samples, features


# In[99]:


mx, ms = points_to_infer[0]
tmp = np.ones((10000, 1))
mx = tmp*mx
ms = tmp*ms
big_arr = np.hstack((mx, ms))

for mx, ms in points_to_infer[1:]:
    tmp = np.ones((10000, 1))
    mx = tmp*mx
    ms = tmp*ms
    horiz_arr = np.hstack((mx, ms))
    big_arr = np.vstack((big_arr, horiz_arr))

big_points_to_infer = np.tile(
    big_arr.reshape(
        len(points_to_infer), 10000, 2,
    ), (10, 1, 1, 1),
)


# In[100]:


flow_data = np.concatenate((flow_samples, big_points_to_infer), axis=-1)

# In[101]:


with torch.no_grad():
    pnn_flow = pytorch_model.forward(
        torch.tensor(
            flow_data, dtype=torch.float32,
        ),
    ).detach().numpy().squeeze()

with uproot.recreate(tree_dir) as file:
    file["samples"] = {f"X{a[0]}_S{a[1]}":pnn_flow[:, i, :] for i,a in enumerate(points_to_infer)}
# In[102]:


# Number of bins for histogram
# bins = 10

# Define the function to apply along the axis




# test_data_hists = []
# with torch.no_grad():
#     for array in test_data:
#         array = ak_to_ndarray(
#             array[['m_jj', 'm_bbyy_mod', 'm_x', 'm_s', 'total_weight']],
#         )
#         pnn_output = pytorch_model.forward(
#             torch.tensor(
#                 array[:, :-1], dtype=torch.float32,
#             ),
#         ).detach().numpy()
#         test_data_hists.append(
#             np.histogram(
#                 pnn_output.ravel(), weights=array[:, -1],
#             ),
#         )
# test_norms = np.array([np.sum(h[0].ravel()) for h in test_data_hists])
# norm_factors = test_norms / num_samples

# def hist_func(x):
#     return np.histogram(x, bins=bins)[0]
# # Apply the function along the third axis (axis=2)
# reduced_array = np.apply_along_axis(
#     hist_func, axis=2, arr=np.squeeze(pnn_flow),
# )
# fhs = reduced_array*np.tile(
#     norm_factors, (10, 1),
# ).reshape(10, len(points_to_infer), 1)
# flow_hists, flow_errs = np.mean(fhs, axis=0), np.std(fhs, axis=0)



## This is all plotting code!

# In[103]:

# def plot_hist(dct):
#     ax, (data, flow, errs), i = dct['ax'], dct['data'], dct['i']
#     # chi2 = chisquare(data[0], f_exp=flow)
#     ax.stairs(data[0], data[1], fill=True, label='signal MC', alpha=0.6)
#     ax.stairs(flow, data[1], label='flow avg')
#     ax.stairs(
#         flow+errs, data[1], linestyle=':', color='k',
#         label=r'flow$\pm$$\sigma_{flow}$', alpha=0.7,
#     )
#     ax.stairs(flow-errs, data[1], linestyle=':', color='k', alpha=0.7)
#     mx, ms = points_to_infer[i]
#     ax.set_title(f'$m_X$, $m_S$ = {mx}, {ms}')  # , {chi2[0]/10:.3g}")
# #     ax.set_xlabel("pNN")
#     ax.legend(fontsize='6', frameon=False, loc='upper left')


# plothelp.autogrid(
#     list(zip(test_data_hists, flow_hists, flow_errs)),
#     plot_hist,
#     subplot_kwargs=dict(facecolor='w', sharex=True),
#     title='pNN hists from flow samples (trained on all)',
# )


# plt.savefig('pnn-interpolation-plots.pdf', bbox_inches='tight')


# # In[ ]:


# # In[ ]:


# # In[ ]:
