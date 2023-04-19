.PHONY: all venv preprocessed cv traditional-ml fourier dwt minirocket tsfresh dl lstm cnn clean 

help: ## Help
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
all: venv preprocessed cv ## Run entire pipeline

#### Build virtual environment and install packages
venv: venv/bin/activate ## Create virtual environment and download required packages from requirements.txt
venv/bin/activate: requirements.txt
	python -m venv venv
	. venv/bin/activate; pip install --upgrade pip; pip install -r requirements.txt; pip install "git+https://github.com/03bennej/ei-python.git"; pip install -e .;

### Preprocess data
preprocessed: data/preprocessed ## Preprocess raw data
data/preprocessed: src/data/preprocessing.py
	rm -rf data/preprocessed 
	mkdir -p data/preprocessed
	venv/bin/python src/data/preprocessing.py --data_path data/raw --test_size 0.2 --split_seed 42 --truncate_seed 111 --save_path data/preprocessed

### Cross validation 
cv: baselines traditional-ml transformations dl ## Run all cross-validation analysis for considered machine learning methods

## CV for traditional ml models
baselines: results/cross-validation/baselines ## Run cross-validation for baselines
results/cross-validation/baselines: 
	rm -rf results/cross-validation/baselines
	mkdir -p results/cross-validation/baselines
	venv/bin/python src/cross-validation/baselines/train.py

## CV for traditional ml models
traditional-ml: results/cross-validation/traditional-ml ## Run cross-validation for traditional machine learning methods
results/cross-validation/traditional-ml:
	rm -rf results/cross-validation/traditional-ml
	mkdir -p results/cross-validation/traditional-ml
	venv/bin/python src/cross-validation/traditional-ml/train.py

## CV for transformations before application of ml
transformations: fourier dwt minirocket tsfresh ## Run cross-validation for time series transformations

fourier: results/cross-validation/transformations/fourier ## Run cross-validation for Fourier transform
results/cross-validation/transformations/fourier: 
	rm -rf results/cross-validation/transformations/fourier
	mkdir -p results/cross-validation/transformations/fourier
	venv/bin/python src/cross-validation/transformations/fourier/train.py

dwt: results/cross-validation/transformations/dwt ## Run cross-validation for discrete wavelet transform
results/cross-validation/transformations/dwt: 
	rm -rf results/cross-validation/transformations/dwt
	mkdir -p results/cross-validation/transformations/dwt
	venv/bin/python src/cross-validation/transformations/dwt/train.py

minirocket: results/cross-validation/transformations/minirocket ## Run cross-validation for MiniRocket transformation
results/cross-validation/transformations/minirocket: 
	rm -rf results/cross-validation/transformations/minirocket
	mkdir -p results/cross-validation/transformations/minirocket
	venv/bin/python src/cross-validation/transformations/minirocket/train.py

tsfresh: results/cross-validation/transformations/tsfresh ## Run cross-validation for tsfresh transformation
results/cross-validation/transformations/tsfresh: 
	rm -rf results/cross-validation/transformations/tsfresh
	mkdir -p results/cross-validation/transformations/tsfresh
	venv/bin/python src/cross-validation/transformations/tsfresh/train.py

## CV for deep learning methods

dl: lstm cnn ## Run cross-validation for deep learning methods

lstm: results/cross-validation/dl/lstm ## Run cross-validation for LSTM
results/cross-validation/dl/lstm: 
	rm -rf results/cross-validation/dl/lstm
	mkdir -p results/cross-validation/dl/lstm
	venv/bin/python src/cross-validation/dl/lstm/ERK_Akt/train.py
	venv/bin/python src/cross-validation/dl/lstm/ERK/train.py
	venv/bin/python src/cross-validation/dl/lstm/Akt/train.py

cnn: results/cross-validation/dl/cnn ## Run cross-validation for CNN
results/cross-validation/dl/cnn: 
	rm -rf results/cross-validation/dl/cnn
	mkdir -p results/cross-validation/dl/cnn
	venv/bin/python src/cross-validation/dl/cnn/ERK_Akt/train.py
	venv/bin/python src/cross-validation/dl/cnn/ERK/train.py
	venv/bin/python src/cross-validation/dl/cnn/Akt/train.py

### Clean

clean-venv: ## Remove virtual environment
	rm -rf venv
	rm -rf src.egg-info
	rm -rf src/__pycache__

clean-all: ## Remove all files
	rm -rf venv
	rm -rf src.egg-info
	rm -rf data/preprocessed
	rm -rf src/__pycache__
	rm -rf src/predicting
	rm -rf results