export CUDA_VISIBLE_DEVICES=1

python train_demo.py \
	--trainN 5 --N 5 --K 1 --Q 1 --max_length 60 \
	--model proto --encoder bert --hidden_size 768 --val_step 1000 \
  --batch_size 4 --alpha 0.0 \
	--path ../../../pretrain_relation/ckpt/ckpt_cp/ckpt_of_step_11000



