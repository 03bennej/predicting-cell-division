preprocessed: src/data/preprocessing.py
	mkdir -p data/preprocessed
	python src/data/preprocessing.py --data_path data/raw --test_size 0.2 --split_seed 42 --truncate_seed 111 --save_path data/preprocessed
