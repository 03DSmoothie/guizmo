# -*- coding: utf-8 -*-
"""Notebook Colab/Kaggle 0 Euro : Guizmo Coeur 150M vrai (T4 gratuit).
VALIDE le 2026-09-10 : OASST FR 249 paires OK sans login, Wikipedia FR OK sans
login, CulturaX = GATED (login requis). Donc sur Colab/Kaggle :
  source auto = Wikipedia FR + OSCAR FR + OASST FR (tout sans login).
Copie ce fichier dans Colab (GPU T4), execution cellule par cellule.
Duree : ~3-6h pretrain + ~1h SFT sur T4. Tout est gratuit.
"""
# %% [markdown]
# # Guizmo 150M vrai - Colab/Kaggle T4 gratuit
# Sang pret : 500 Vie TON ton + 200 Raison pote + 249 OASST FR (verifie).
# Pretrain : Wikipedia FR + OSCAR FR (~1Go, sans login).
# Tokenizer 32k -> Pretrain 150M (20k steps) -> SFT 60/30/10 -> push HF

# %% Cell 0 : repo + GPU
# !nvidia-smi
# !git clone https://github.com/TONUSER/guizmo.git
# %cd guizmo
# !pip install -q -r requirements.txt

# %% Cell 1 : datasets reels SANS login (~300Mo, ~20 min, VALIDE)
# !python -m scripts.build_datasets --with-hf --hf-gb 1.0 --max-docs 300000 --source auto

# %% Cell 1b (optionnel, si tu veux CulturaX gated) :
# !huggingface-cli login  # colle ton token HF (regle les conditions uonlp/CulturaX)
# !python -m scripts.build_datasets --with-hf --hf-gb 1.0 --max-docs 300000 --source culturax

# %% Cell 2 : tokenizer 32k sur vrai corpus (~5 min)
# !python -m scripts.train_tokenizer --corpus "data/processed/pretrain.txt" "data/raw/hf_fr.txt" --out tokenizer --vocab_size 32000

# %% Cell 3 : PRETRAIN 150M vrai (~4h T4)
# !python -m scripts.train_pretrain --tokenizer tokenizer --train data/processed/pretrain.txt --block_size 512 --batch_size 4 --grad_accum 8 --lr 3e-4 --max_steps 20000 --device cuda

# %% Cell 4 : SFT Coeur 60/30/10 (~1h)
# !python -m scripts.train_sft --tokenizer tokenizer --epochs 3 --batch_size 4 --max_len 512 --from_ckpt checkpoints/guizmo-150m-pretrain/final.pt --device cuda

# %% Cell 5 : eval + push HF
# !python -m scripts.eval --ckpt checkpoints/guizmo-150m-coeur/coeur-ep3.pt --tokenizer tokenizer --device cuda
# from huggingface_hub import HfApi
# api = HfApi()
# api.create_repo("TONUSER/guizmo-150m-coeur", exist_ok=True)
# api.upload_folder(folder_path="checkpoints/guizmo-150m-coeur", repo_id="TONUSER/guizmo-150m-coeur")
# api.upload_folder(folder_path="tokenizer", repo_id="TONUSER/guizmo-150m-coeur")
