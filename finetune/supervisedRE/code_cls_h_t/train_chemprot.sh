export CUDA_VISIBLE_DEVICES=1

python main.py \
	--lr 3e-5 --batch_size_per_gpu 64 --max_epoch 100 --model cls \
	--max_length 100 \
	--mode CM \
	--dataset chemprot \
	--entity_marker --ckpt_to_load ../../../pretrain_relation/ckpt/ckpt_cp/ckpt_of_step_11000 \
	--train_prop 1.0

python main.py \
	--lr 3e-5 --batch_size_per_gpu 64 --max_epoch 100 --model cls \
	--max_length 100 \
	--mode CM \
	--dataset chemprot \
	--entity_marker --ckpt_to_load ../../../pretrain_relation/ckpt/ckpt_cp/ckpt_of_step_11000 \
	--train_prop 0.1

python main.py \
	--lr 3e-5 --batch_size_per_gpu 64 --max_epoch 100 --model cls \
	--max_length 100 \
	--mode CM \
	--dataset chemprot \
	--entity_marker --ckpt_to_load ../../../pretrain_relation/ckpt/ckpt_cp/ckpt_of_step_11000 \
	--train_prop 0.01
