# predicting-cell-division

The following steps can be run on a *nix machine to reproduce our work:

1) Download and install python.
2) Fork the project repo and navigate to it.
3) Type `make` to build entire project, though this will take some time. Instead you can run step by step.
4) Type `make venv` to create a virtual environment and download all required python packages.
5) Type `make preprocessed` to preprocess the raw data for classification.
6) Type `make cv` to run all cross validation analysis.
6) Type `make preprocessed` to preprocess the raw data. In `Makefile`, edit `split_seed` for a different train/test split, or `truncate_seed` for a different sampling of truncated time points.