# Interpolation-with-flows
For the Run-2 SH->bbyy analysis in ATLAS!

## Intro
There is a script for each of the following tasks:
- `load_files.py`: Pre-processing ntuples into awkward arrays & applying pre-selection
- `train-flow.py`: Training loop for flow model(s)
- `generate-flow-samples.py`: Sampling points from trained flow models conditioned on a particular signal grid point, then applying the trained pNN model on the outputs to get a pNN shape. Writes out TTrees into a root file.
- `convert-to-onnx.py`: Converting a Keras pNN model to Pytorch so that it can be applied without installing Tensorflow (but you do need Keras to run that script!)

There is also a folder `flow_interp/` that contains the core code for the flow methods, and `flow-weights-all/`, which has the most recent flows I trained -- 10 flows on the whole mass grid (without the new random points).

## Setup
In your environment (preferably a fresh virtual env), install the dependencies:

`python3 -m pip install -r requirements.txt`

Note: this will install the CPU version of PyTorch, and may take up a good amount of space (100s of MBs). If you want to convert a Keras model with`convert-to-onnx.py`, you will also need Keras installed (which may install tensorflow -- another large package).

All scripts use the same config file `config.json`. The important options for each script should be loaded at the start of the script, which should hopefully make clear which ones you need to modify (ask me if unsure).

The main thing I've tested is generating flow samples. To do this with the existing flows, you just need to run `load_files.py` (which needs the path to our current ntuples as the `data_path` field in `config.json`). The output file name and path is controlled by `sig_arr_path`, which is also where the other scripts will look to find the preprocessed signal MC array you just generated. 

After that, you can generate samples! Here are the important config options for `generate-flow-samples.py`:

```
"onnx_model": Path to the onnx-converted model of the pNN (outputted from convert-to-onnx.py)
"path_to_flows": Path to folder containing the trained flow models.
"points_to_infer": The desired signal points to generate samples from in the form [mX, mS]. An example:
[
    [210, 70],
    [245, 90],
    [190, 50],
    [300, 110],
    [500, 170],
    [750, 400]
],
"feature_scaler_path": Path to .bin file containing the scaler for mbbyy* and mbb.
"context_scaler_path": Path to .bin file containing the scaler for the truth mass points.
"num_samples": How many samples to generate. I've been using 10,000 as a baseline, then normalizing the resulting histograms appropriately.
"samples_filename": Name of the root file containing the samples.
```

Then, just run `python3 generate-flow-samples.py`. The result will give you a TTree indexed by each signal point, and will have an array of size [number of flows, number of samples]. You can get the final histograms by averaging the histograms of all 10 flows, and use the standard deviation as a shape systematic.
