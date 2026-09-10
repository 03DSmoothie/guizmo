# DEMARRAGE GUIZMO (Windows, 0 EUR)

## 1. Comprendre (ton README)
- **Coeur** = petit LLM FR 150M (12 couches, 768 hidden, GQA 12Q/4KV, RoPE, SwiGLU, RMSNorm) — style Llama3/Qwen3/SmolLM2. Il apprend a parler + raisonner, PAS le monde.
- **Yeux** = `guizmo/search.py` (DuckDuckGo gratuit). Le Coeur emet `<search>requete</search>`, on injecte `<result>`, il re-ecrit avec son avis.
- **Mix SFT impose** : Vie 60% / Raison 30% / Chercheur 10% (`guizmo/data.py::mix_datasets`).

## 2. Installer (1 fois)
```powershell
pip install -r requirements.txt
```

## 3. Pipeline complet gratuit
```powershell
python -m scripts.build_datasets
python -m scripts.train_tokenizer --corpus "data/processed/pretrain.txt" --out tokenizer --vocab_size 32000
python -m scripts.train_pretrain --tokenizer tokenizer --train data/processed/pretrain.txt --tiny --max_steps 2000
python -m scripts.train_sft --tokenizer tokenizer --tiny --epochs 3 --from_ckpt checkpoints/guizmo-150m-pretrain/final.pt
python -m scripts.eval --ckpt checkpoints/guizmo-150m-coeur/coeur-ep3.pt --tokenizer tokenizer
python -m scripts.chat --ckpt checkpoints/guizmo-150m-coeur/coeur-ep3.pt --tokenizer tokenizer
python app.py
pytest -q
```

## 4. Passer au vrai 150M + vrais FR
- Remplace `--tiny` par rien (150M), `--max_steps 20000`, `--block_size 512`, GPU Colab/Kaggle T4 gratuit.
- Vrais datasets FR a brancher dans `scripts/build_datasets.py` :
  - Vie : `OpenAssistant/oasst1` filtre `lang==fr` + 500 dialogues TON ton (fichier `scripts/seed_data.py` a grossir).
  - Pretrain FR : `Helsinki-NLP/uncorpus` fr / `oscar-corpus/OSCAR-2301` fr / `CulturaX` fr (filtre qualite).
- Tokenizer 32k -> garde les tokens `<user> <guizmo> <search> </search> <result> </result>`.

## 5. Tests
`tests/test_guizmo.py` : forward, RMSNorm, mix 60/30/10, detection `<search>`.

## Structure
```
configs/guizmo_150m.yaml  guizmo/config.py guizmo/model.py guizmo/tokenizer.py guizmo/data.py guizmo/search.py guizmo/converse.py
scripts/build_datasets.py scripts/seed_data.py scripts/train_tokenizer.py scripts/train_pretrain.py scripts/train_sft.py scripts/eval.py scripts/chat.py
app.py  tests/  requirements.txt
```
