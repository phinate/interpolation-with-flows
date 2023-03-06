# Interpolation-with-flows
For the Run-2 SH->bbyy analysis in ATLAS!

## Usage
There is a script for the following tasks:
- Pre-processing ntuples into awkward arrays & applying pre-selection
- Training loop for flow model(s)
- Sampling points from trained flow models conditioned on a particular signal grid point, then applying the trained pNN model on the outputs to get a pNN shape
- Converting a Keras pNN model to Pytorch so that it can be applied without installing Tensorflow (but you do need Keras to run that script!)