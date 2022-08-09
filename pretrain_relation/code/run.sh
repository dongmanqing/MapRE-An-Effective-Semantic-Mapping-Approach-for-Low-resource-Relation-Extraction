#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node 4  main.py \
#	--cuda 4,5,6,7 \
#        --model MTB \
#	--lr 3e-5 --batch_size_per_gpu 32 --max_epoch 20 \
#	--gradient_accumulation_steps 2 \
#	--max_length 64 \
#	--save_step 5000 \
#	--alpha 0.3 \
#	--temperature 0.05 \
#	--train_sample \
#	--save_dir ckpt_mtb \

python prepare_data.py

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

python -m torch.distributed.launch --nproc_per_node 6 --master_port 29501 main.py \
	--cuda 2,3,4,5,6,7 \
	--model CP \
	--lr 3e-5 --batch_size_per_gpu 170 --max_epoch 60 \
	--gradient_accumulation_steps 2 \
	--max_length 60 \
	--save_step 500 \
	--alpha 0.3 \
	--temperature 0.05 \
	--train_sample \
	--save_dir ckpt_cp
