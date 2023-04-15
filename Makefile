.PHONY: all venv preprocessed cv traditional-ml dwt clean

all: venv preprocessed cv

### Build virtual environment and install packages
venv: venv/bin/activate
venv/bin/activate: requirements.txt
	python -m venv venv
	. venv/bin/activate; pip install -r requirements.txt; pip install "git+https://github.com/03bennej/ei-python.git"; pip install -e .;

### Preprocess data
preprocessed: data/preprocessed
data/preprocessed: src/data/preprocessing.py
	rm -rf data/preprocessed 
	mkdir -p data/preprocessed
	venv/bin/python src/data/preprocessing.py --data_path data/raw --test_size 0.2 --split_seed 42 --truncate_seed 111 --save_path data/preprocessed

### Cross validation
cv: traditional-ml transformations

## CV for traditional ml models
traditional-ml: results/cross-validation/traditional-ml
results/cross-validation/traditional-ml: data/preprocessed src/cross-validation/traditional-ml/train.py src/ei_setup.py
	rm -rf results/cross-validation/traditional-ml
	mkdir -p results/cross-validation/traditional-ml
	venv/bin/python src/cross-validation/traditional-ml/train.py

## CV for transformations before ml
transformations: dwt

# dwt
dwt: results/cross-validation/transformations/dwt
results/cross-validation/transformations/dwt: data/preprocessed src/cross-validation/transformations/dwt/train.py src/ei_setup.py
	rm -rf results/cross-validation/transformations/dwt
	mkdir -p results/cross-validation/transformations/dwt
	venv/bin/python src/cross-validation/transformations/dwt/train.py

### Remove all created files
clean:
	rm -rf venv
	rm -rf src.egg-info
	rm -rf data/preprocessed
	rm -rf src/__pycache__
	rm -rf src/predicting
	rm -rf results


