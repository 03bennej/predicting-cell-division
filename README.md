# Predicting cell division using ERK and Akt kinase activity time courses

The following instructions can be run on a *nix machine to reproduce our work:

>1) Download and install python.
>2) Fork the project repo and navigate to it on your local machine. Typing `make help` gives a complete list of make commands that are run consecutively.
>3) Type `make all` to run all make commands consecutively, though this will take some time. Instead you can run step-by-step.
>> a) Type `make venv` to create a virtual environment and download all required python packages.
>> b) Type `make preprocessed` to preprocess the raw data for classification. In `Makefile`, edit `split_seed` for a different train/test split, or `truncate_seed` for a different sampling of truncated time points.
>> c) Type `make cv` to run a cross validation of all considered methods.
>> d) Type `make final-models` to train final models. 
>> e) Type `make test` to test models on test sets.