# Predicting cell division using ERK and Akt kinase activity time courses

The following instructions can be run on a *nix machine to reproduce our work:

1. Download and install Python. Our results were produced using Python 3.11.5, and the specific versions of packages in ``requirements.txt`` may require it, but you may try with another Python version...

2. Fork the project repo and navigate to it on your local machine. Typing `make help` gives a complete list of make commands to be run consecutively.

3. Type `make all` to run all make commands consecutively, though this will take some time. Instead you can run step-by-step:
    * Type `make venv` to create a virtual environment and download all required python packages. 
    * Type `make processed` to process the raw data for classification. In `Makefile`, edit `split_seed` for a different train/test split, or `truncate_seed` for a different sampling of truncated time points.
    * Cross validated performance analysis can be run for a number of individual methods, e.g. `make lstm` analyses the LSTM method. Inspect the Makefile, or type `make help` for more commands.
    * Type `make models` to train final ensemble models. 
    * Type `make interpretation` to run the interpretation algorithm.
    * Type `make test` to test models on test sets.

4. Results are saved in the ``results/`` folder. We provide notebooks that provide visualizations and further analysis in ``notebooks/``.