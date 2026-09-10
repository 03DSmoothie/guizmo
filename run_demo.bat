@echo off
pip install -r requirements.txt
python -m scripts.build_datasets
python -m scripts.train_tokenizer --corpus "data/processed/pretrain.txt" --out tokenizer --vocab_size 32000
python -m scripts.train_pretrain --tokenizer tokenizer --train data/processed/pretrain.txt --tiny --max_steps 200
python -m scripts.train_sft --tokenizer tokenizer --tiny --epochs 1 --from_ckpt checkpoints/guizmo-150m-pretrain/final.pt
pytest -q
pause
