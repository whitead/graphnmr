# Graph NMR

A graph convolutional neural network for predicting NMR chemical shifts in
molecules.

## Layout

* `data`: The TF Records
* `graphnmr`: The installable module containing model code
  * `__init__.py`: The module init file
  * `data.py`: functions for processing and loading data
  * `gcnmodel.py`: The main model code
  * `validation.py`: Validation code for checking correctness of data
* `parse`: Scripts for converting raw data into TF records for training
* `scripts`: Scripts for running model
  * `plot_gcn_comparison.py`: Script for plotting hyperparameters choices on grid
  * `train_hypers.py`: Script for running with variety of hyperparameters
  * `train_structural.py`: Main training script

## Data

The raw data is not in this repo due to the huge number of files.

## Preequisites

numpy, matplotlib, tensorflow pre 2.0, graphviz, networkx, tqdm, gsd (conda-forge). If you want to do the data processing stuff,
use the yml file included.