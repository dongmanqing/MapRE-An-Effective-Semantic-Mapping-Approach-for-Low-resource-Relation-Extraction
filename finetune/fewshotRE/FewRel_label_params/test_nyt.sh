export CUDA_VISIBLE_DEVICES=5,6,7

python train_demo.py --test test_nyt --alpha 0.1 \
	--trainN 5 --N 5 --K 1 --Q 1 --max_length 200 \
	--model proto --encoder bert --hidden_size 768 --val_step 100 --test_iter 200 \
  --batch_size 100 --only_test --load_ckpt checkpoint/proto-bert-train_nyt-val_nyt-5-5-alpha_0.1.pth.tar

python train_demo.py --test test_nyt --alpha 0.1 \
	--trainN 5 --N 5 --K 5 --Q 1 --max_length 200 \
	--model proto --encoder bert --hidden_size 768 --val_step 100 --test_iter 200 \
  --batch_size 100 --only_test --load_ckpt checkpoint/proto-bert-train_nyt-val_nyt-5-5-alpha_0.1.pth.tar

python train_demo.py --test test_nyt --alpha 0.1 \
	--trainN 5 --N 10 --K 1 --Q 1 --max_length 200 \
	--model proto --encoder bert --hidden_size 768 --val_step 100 --test_iter 200 \
  --batch_size 100 --only_test --load_ckpt checkpoint/proto-bert-train_nyt-val_nyt-5-5-alpha_0.1.pth.tar

python train_demo.py --test test_nyt --alpha 0.1 \
	--trainN 5 --N 10 --K 5 --Q 1 --max_length 200 \
	--model proto --encoder bert --hidden_size 768 --val_step 100 --test_iter 200 \
  --batch_size 100 --only_test --load_ckpt checkpoint/proto-bert-train_nyt-val_nyt-5-5-alpha_0.1.pth.tar
