export CUDA_VISIBLE_DEVICES=1

for i in $(seq 2 10)
do python train_demo.py --test test_nyt --alpha 0.0 --test_iter 200 \
	--trainN $i --N $i --K 1 --Q 1 --max_length 200 \
	--model proto --encoder bert --hidden_size 768 --val_step 1000 \
  --batch_size 300 --only_test --load_ckpt checkpoint/proto-bert-train_nyt-val_nyt-5-1-alpha_0.0.pth.tar
done

