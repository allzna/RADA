root_path=./dataset

python -u run.py \
--mode test \
--configs_path ./configs/ \
--save_path ./test_results/ \
--root_path $root_path \
--data SMD \
--data_origin DADA \
--gpu 0