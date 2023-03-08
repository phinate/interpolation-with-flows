import os
from glob import glob
import uproot
import awkward as ak
import numpy as np


def get_data(process="signal", fields=("m_jj", "m_bbyy", "m_x", "m_s", "total_weight"), save = False):
    data_path = "/eos/user/s/sandrean/ntuples/SH/h027"

    if process == "signal":

        paths = glob(os.path.join(data_path, "X*.root"))

        arrs = []

        for filename in paths:
            dct = {}
            arr = uproot.open(filename)["CollectionTree;1"].arrays()
            name = filename.replace(".root", "")
            name = name.replace(data_path, "")
            x_idx = name.find("X")
            s_idx = name.find("S")

            X_mass = ak.Array(
                np.zeros(len(arr)) + int(name[x_idx + 1 : s_idx - 1]),
            )
            S_mass = ak.Array(
                np.zeros(len(arr)) + int(name[s_idx + 1 :]),
            )

            arrs.append(ak.with_field(ak.with_field(arr, X_mass, "m_x"), S_mass, "m_s"))

        if save:
            ak.to_parquet(ak.concatenate(arrs)[list(fields)], "signals.parquet")
        else:
            return ak.concatenate(arrs)[list(fields)]

    elif process == "background":

        paths = glob(os.path.join(data_path, "*.root"))

        arrs = []

        for filename in paths:
            dct = {}
            arr = uproot.open(filename)["CollectionTree;1"].arrays()
            name = filename.replace(".root", "")
            name = name.replace(data_path, "")
            x_idx = name.find("X")
            s_idx = name.find("S")

            X_mass = ak.Array(
                np.zeros(len(arr)) + int(name[x_idx + 1 : s_idx - 1]),
            )
            S_mass = ak.Array(
                np.zeros(len(arr)) + int(name[s_idx + 1 :]),
            )

            arrs.append(ak.with_field(ak.with_field(arr, X_mass, "m_x"), S_mass, "m_s"))

        if save:
            ak.to_parquet(ak.concatenate(arrs)[list(fields)], "backgrounds.parquet")
        else:
            return ak.concatenate(arrs)[list(fields)]