# DEMARRAGE GUIZMO (Windows / Kaggle / Colab, 0 EUR)

## 0. V3 — LA VOIE PRINCIPALE (post-crash Kaggle, refonte 2026)
Kaggle a crashé sur le pretrain 150M. Résultat : on a RÉUSSI la refonte encore
plus légère que prévu. Guizmo ne stocke AUCUNE connaissance du monde.

- **Nano (~10M)** : 4 couches / 384 hidden / GQA 6Q+2KV / SwiGLU — il apprend
  UNIQUEMENT à lire la question et sortir le format
  `<route> intention | requete1 ; requete2 </route>`.
  Il comprend **INTENTION + CONTEXTE + QUESTION**, c'est tout.
- **Yeux** : `guizmo/search.py` — DDG + fallback Wikipedia FR (sans clé).
  Guizmo **cherche à chaque fois** sur internet : le savoir vient du web, jamais de sa mémoire.
- **Synthèse** : `guizmo/pipeline.py` — faits digérés + avis + relance, ton pote.
  Sans modèle chargé : synthèse extractive (il ne recrache que ce que le web a dit,
  jamais d'invention). Salutation / émotion pure → réponse directe (rien à chercher).

```powershell
python -m scripts.build_router                                # dataset routage (>=787 ex)
python -m scripts.train_router --nano --epochs 2              # ~2 min sur CPU
python -m scripts.chat_v2                                     # CLI V3 (règles)
python -m scripts.chat_v2 --routeur checkpoints/guizmo-nano-routeur/routeur-ep2.pt
python app.py                                                 # démo web Gradio V3
python -m pytest tests -q                                     # 7 passed
```
Le 150M (Coeur complet) reste optionnel en V1 : voir `colab_guizmo_150m.py`.

---


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
guizmo/config.py nano()~10M + coeur150m()   guizmo/model.py (RMSNorm+RoPE+GQA+SwiGLU)
guizmo/router.py routeur intention/contexte/requetes   guizmo/search.py DDG+Wiki
guizmo/pipeline.py route->search->synthese   guizmo/synth.py prompts+fallbacks
scripts/build_router.py scripts/train_router.py scripts/chat_v2.py   app.py (V3 web)
scripts/build_datasets.py train_tokenizer/pretrain/sft/eval.py (V1 150M optionnel)
tests/test_guizmo.py  data/seed/routeur.jsonl
```
