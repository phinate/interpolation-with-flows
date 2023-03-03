#!/usr/bin/env python
# In[2]:
from __future__ import annotations

import os
from glob import glob

import awkward as ak
import numpy as np
import uproot


# In[3]:

outdir = ''
data_path = '/eos/user/h/hhsukth/GroupSpace/Ntuples'
paths = glob(os.path.join(data_path, 'X*.root'))

arrs = []
pre = '(m_yy > 105) & (m_yy < 160) & (cutFlow == 7)'
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
