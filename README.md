# Predicting cell division using ERK and Akt kinase activity time courses

The following instructions can be run on a *nix machine to reproduce our work:

1. Download and install python.

2. Fork the project repo and navigate to it on your local machine. Typing `make help` gives a complete list of make commands to be run consecutively.

3. Type `make all` to run all make commands consecutively, though this will take some time. Instead you can run step-by-step:
    * Type `make venv` to create a virtual environment and download all required python packages. 
    * Type `make preprocessed` to preprocess the raw data for classification. In `Makefile`, edit `split_seed` for a different train/test split, or `truncate_seed` for a different sampling of truncated time points.
    * Type `make cv` to run a cross validation of all considered methods. Cross validation can be run for individual methods instead, e.g. `make dl` performs cross validation on all deep learning methods, whereas `make transformations` performs cross validation on all transformations, whereas `make dwt` performs cross validation on the DWT method only. See `make help` for more commands.
    * Type `make final-models` to train final models. 
    * Type `make test` to test models on test sets.

4. Cross validation and test results can be found in the "results" folder. 

5. Visualisations can be found in various notebooks in the "notebooks" folder.

6. Final models can be found in "models". They are all heterogeneous ensembles generated using Ensemble Integration (EI). We refer the user to the [EI github repo](https://github.com/GauravPandeyLab/ei-python) for instructions on their use.