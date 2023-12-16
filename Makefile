.PHONY: all venv preprocessed cv traditional_ml fourier dwt minirocket tsfresh dl lstm cnn clean 

help: ## Help
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
all: venv preprocessed cv models ## Run entire pipeline

#### Build virtual environment and install packages ####
venv: venv/bin/activate ## Create virtual environment and download required packages from requirements.txt
venv/bin/activate: requirements.txt
	python -m venv venv
	. venv/bin/activate; pip install --upgrade pip; pip install -r requirements.txt; pip install -e .;

#### Preprocess data ####
processed: data/processed  ## Preprocess raw data
data/processed: src/data/processing_mcf10a.py
	rm -rf data/processed 
	mkdir -p data/processed
	venv/bin/python src/data/processing_mcf10a.py --data_path data/raw/mcf10a --test_size 0.2 --split_seed 42 --truncate_seed 111 --save_path data/processed/mcf10a
	venv/bin/python src/data/processing_rpe.py

#### Cross validation of methods ####

# CV for xgboost
xgboost: results/cross_validation/xgboost ## Run cross_validation for baselines
results/cross_validation/xgboost: 
	rm -rf results/cross_validation/xgboost
	mkdir -p results/cross_validation/xgboost
	venv/bin/python src/cross_validation/xgboost/train.py

# CV for traditional ml models
traditional_ml: results/cross_validation/traditional_ml ## Run cross_validation for traditional machine learning methods
results/cross_validation/traditional_ml:
	rm -rf results/cross_validation/traditional_ml
	mkdir -p results/cross_validation/traditional_ml
	venv/bin/python src/cross_validation/traditional_ml/train.py

# CV for transformations before application of ml
transformations: fourier dwt minirocket tsfresh ## Run cross_validation for time series transformations

fourier: results/cross_validation/transformations/fourier ## Run cross_validation for Fourier transform
results/cross_validation/transformations/fourier: 
	rm -rf results/cross_validation/transformations/fourier
	mkdir -p results/cross_validation/transformations/fourier
	venv/bin/python src/cross_validation/transformations/fourier/train.py

dwt: results/cross_validation/transformations/dwt ## Run cross_validation for discrete wavelet transform
results/cross_validation/transformations/dwt: 
	rm -rf results/cross_validation/transformations/dwt
	mkdir -p results/cross_validation/transformations/dwt
	venv/bin/python src/cross_validation/transformations/dwt/train.py

minirocket: results/cross_validation/transformations/minirocket ## Run cross_validation for MiniRocket transformation
results/cross_validation/transformations/minirocket: 
	rm -rf results/cross_validation/transformations/minirocket
	mkdir -p results/cross_validation/transformations/minirocket
	venv/bin/python src/cross_validation/transformations/minirocket/train.py

tsfresh: results/cross_validation/transformations/tsfresh ## Run cross_validation for tsfresh transformation
results/cross_validation/transformations/tsfresh: 
	rm -rf results/cross_validation/transformations/tsfresh
	mkdir -p results/cross_validation/transformations/tsfresh
	venv/bin/python src/cross_validation/transformations/tsfresh/train.py

#### CV for deep learning methods

lstm: results/cross_validation/dl/lstm # Run cross_validation for LSTM
results/cross_validation/dl/lstm: 
	rm -rf results/cross_validation/dl/lstm
	mkdir -p results/cross_validation/dl/lstm
	venv/bin/python src/cross_validation/dl/lstm/ERK_Akt/train.py
	venv/bin/python src/cross_validation/dl/lstm/ERK/train.py
	venv/bin/python src/cross_validation/dl/lstm/Akt/train.py

cnn: results/cross_validation/dl/cnn # Run cross_validation for CNN
results/cross_validation/dl/cnn: 
	rm -rf results/cross_validation/dl/cnn
	mkdir -p results/cross_validation/dl/cnn
	venv/bin/python src/cross_validation/dl/cnn/ERK_Akt/train.py
	venv/bin/python src/cross_validation/dl/cnn/ERK/train.py
	venv/bin/python src/cross_validation/dl/cnn/Akt/train.py

#### Build final models

models: results/models ## Build final models
results/models:
	rm -rf results/models
	mkdir -p results/models
	venv/bin/python src/models/ERK_Akt/train.py
	venv/bin/python src/models/ERK/train.py
	venv/bin/python src/models/Akt/train.py

#### Test final models on test datasets

test: high_dose low_dose rpe ## Test final models on test sets

high_dose: results/testing/mcf10a/high_dose
results/testing/mcf10a/high_dose:
	rm -rf results/testing/mcf10a/high_dose
	mkdir -p results/testing/mcf10a/high_dose
	venv/bin/python src/testing/mcf10a/high_dose/test.py

low_dose: results/testing/mcf10a/low_dose
results/testing/mcf10a/low_dose:
	rm -rf results/testing/mcf10a/low_dose
	mkdir -p results/testing/mcf10a/low_dose
	venv/bin/python src/testing/mcf10a/low_dose/test.py
	
rpe: results/testing/rpe
results/testing/rpe:
	rm -rf results/testing/rpe
	mkdir -p results/testing/rpe
	venv/bin/python src/testing/rpe/test.py

#### Interpretation of ERK/Akt model

interpretation: results/interpretation
results/interpretation:
	rm -rf results/interpretation
	mkdir -p results/interpretation
	venv/bin/python src/interpretation/interpretation.py

#### Clean

clean-venv: ## Remove virtual environment
	rm -rf venv
	rm -rf src.egg-info
	rm -rf src/__pycache__

clean-all: ## Remove all files
	rm -rf venv
	rm -rf src.egg-info
	rm -rf data/processed
	rm -rf src/__pycache__
	rm -rf results