#!/usr/bin/env python

from __future__ import annotations

import json
import os
from glob import glob

import awkward as ak
import numpy as np
import uproot


# Load config details from JSON file
with open('config.json', 'r') as f:
    config = json.load(f)

data_path = config['data_path']
outdir = config['sig_arr_folder']
pre = config['preselection']

# Find root files to load
paths = glob(os.path.join(data_path, 'X*.root'))

arrs = []
for filename in paths:
    dct = {}
    arr = uproot.open(filename)['CollectionTree;1'].arrays(cut=pre)
    name = filename.replace('.root', '')
    name = name.replace(data_path, '')
    x_idx = name.find('X')
    s_idx = name.find('S')

    X_mass = ak.Array(
        np.zeros(len(arr)) + int(name[x_idx + 1: s_idx - 1]),
    )
    S_mass = ak.Array(
        np.zeros(len(arr)) + int(name[s_idx + 1:]),
    )

    arrs.append(
        ak.with_field(
            ak.with_field(
                arr, X_mass, 'm_x',
            ), S_mass, 'm_s',
        ),
    )

all_t = ak.concatenate(arrs)
ak.to_parquet(all_t, os.path.join(outdir, 'signals-pnn.parquet'))
