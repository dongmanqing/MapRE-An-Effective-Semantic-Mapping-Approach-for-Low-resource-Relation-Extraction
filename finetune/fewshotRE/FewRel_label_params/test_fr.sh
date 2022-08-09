export CUDA_VISIBLE_DEVICES=0,1,3,2

python train_demo.py --test test_wiki_input-5-1 --alpha 0.1 \
	--trainN 5 --N 5 --K 1 --Q 1 --max_length 60 \
	--model proto --encoder bert --hidden_size 768 --val_step 1000 \
  --batch_size 300 --only_test --load_ckpt checkpoint/proto-bert-train_wiki-val_wiki-5-1-alpha_0.1.pth.tar

python train_demo.py --test test_wiki_input-5-5 --alpha 0.1 \
	--trainN 5 --N 5 --K 5 --Q 5 --max_length 60 \
	--model proto --encoder bert --hidden_size 768 --val_step 1000 \
  --batch_size 300 --only_test --load_ckpt checkpoint/proto-bert-train_wiki-val_wiki-5-1-alpha_0.1.pth.tar

python train_demo.py --test test_wiki_input-10-1 --alpha 0.1 \
	--trainN 10 --N 10 --K 1 --Q 1 --max_length 60 \
	--model proto --encoder bert --hidden_size 768 --val_step 1000 \
  --batch_size 300 --only_test --load_ckpt checkpoint/proto-bert-train_wiki-val_wiki-5-1-alpha_0.1.pth.tar

python train_demo.py --test test_wiki_input-10-5 --alpha 0.1 \
	--trainN 10 --N 10 --K 5 --Q 5 --max_length 60 \
	--model proto --encoder bert --hidden_size 768 --val_step 1000 \
  --batch_size 300 --only_test --load_ckpt checkpoint/proto-bert-train_wiki-val_wiki-5-1-alpha_0.1.pth.tar

