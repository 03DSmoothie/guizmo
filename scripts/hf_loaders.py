"""Chargeurs 0 EUR : corpus FR ouverts SANS login (streaming) + OASST FR.
Probleme trouve sur ton PC : CulturaX uonlp = gated (login requis),
cache HF bloque (WinError 5). Fix :
  - HF_HOME force vers ./hf_cache du projet (droits OK)
  - Sources SANS login : Wikipedia FR + OSCAR-2301 FR (non-gated,
    parquet par langue) + OASST1/OASST2 FR (+ fallback oasst2 FR curated).
Filtre qualite : longueur, francais, pas de spam. Cache local data/raw/.
"""
import os
import re
from pathlib import Path

# Force un cache local avec droits d'ecriture (fix WinError 5 sur C:/Users/.cache)
_PROJ = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(_PROJ / ".hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(_PROJ / ".hf_cache" / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(_PROJ / ".hf_cache" / "datasets"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

MIN_LEN = 80
MAX_LEN = 8000
SPAM_RE = re.compile(r"(http://|https://|www\.|@\w+|<script|viagra|casino|XXX)", re.I)


def is_clean_fr(text: str) -> bool:
    if not text or not (MIN_LEN <= len(text) <= MAX_LEN):
        return False
    if SPAM_RE.search(text):
        return False
    fr_marks = (" le ", " la ", " les ", " de ", " des ", " et ", " est ", " que ", " pas ", "é", "è", "à", "ç")
    low = " " + text.lower() + " "
    return sum(1 for m in fr_marks if m in low) >= 2


def load_wikipedia_fr(limit_chars: float = 1.0e9, max_docs: int = 300000):
    """Wikipedia FR (wikimedia/wikipedia 20231101.fr) : propre, FR natif, non-gated."""
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True)
    print("Corpus : wikimedia/wikipedia 20231101.fr")
    out, size = [], 0
    for ex in ds:
        t = ex.get("text", "")
        if is_clean_fr(t):
            out.append(t)
            size += len(t.encode("utf-8", "ignore"))
            if len(out) >= max_docs or size >= limit_chars:
                break
    print(f"Wikipedia FR : {len(out)} docs, {size/1e6:.1f} Mo")
    return out


def load_oscar_fr(limit_chars: float = 1.0e9, max_docs: int = 300000):
    """OSCAR-2301 FR (oscar-corpus/OSCAR-2301, config fr, non-gated) : gros volume FR."""
    from datasets import load_dataset
    tried = [("oscar-corpus/OSCAR-2301", "fr"), ("Helsinki-NLP/uncorpus", "fr")]
    last_err = None
    for name, cfg in tried:
        try:
            ds = load_dataset(name, cfg, split="train", streaming=True, trust_remote_code=False)
            print(f"Corpus : {name}/{cfg}")
            out, size = [], 0
            for ex in ds:
                t = ex.get("text", "")
                if is_clean_fr(t):
                    out.append(t)
                    size += len(t.encode("utf-8", "ignore"))
                    if len(out) >= max_docs or size >= limit_chars:
                        break
            print(f"OSCAR FR : {len(out)} docs, {size/1e6:.1f} Mo")
            return out
        except Exception as e:
            last_err = e
    raise RuntimeError(f"OSCAR introuvable : {last_err}")


def load_culturax_fr(limit_gb: float = 1.0, split: str = "fr", max_docs: int = 300000):
    """CulturaX uonlp = GATED (il faut `huggingface-cli login` + accepter les conditions).
    Sur Colab/Kaggle : fais le login puis relance. Sinon utilise Wikipedia/OSCAR ci-dessus.
    On garde le nom pour compat avec build_datasets --with-hf."""
    from datasets import load_dataset
    ds = load_dataset("uonlp/CulturaX", split, streaming=True)  # leve une erreur claire si pas logge
    print(f"CulturaX : uonlp/CulturaX/{split}")
    out, size = [], 0
    for ex in ds:
        t = ex.get("text", "")
        if is_clean_fr(t):
            out.append(t)
            size += len(t.encode("utf-8", "ignore"))
            if len(out) >= max_docs or size >= limit_gb * 1e9:
                break
    print(f"CulturaX FR : {len(out)} docs, {size/1e6:.1f} Mo")
    return out


def load_oasst2_fr():
    """OpenAssistant oasst2 (non-gated) : threads FR assistant bien note. Fallback oasst1."""
    from datasets import load_dataset
    try:
        ds = load_dataset("OpenAssistant/oasst2", split="train")
    except Exception:
        ds = load_dataset("OpenAssistant/oasst1", split="train")
    by_tree: dict = {}
    for ex in ds:
        if ex.get("lang") != "fr":
            continue
        tid = ex.get("message_tree_id")
        by_tree.setdefault(tid, []).append(ex)
    rows = []
    for tid, msgs in by_tree.items():
        msgs.sort(key=lambda m: (m.get("created_date") or ""))
        cur_q = None
        for m in msgs:
            if m.get("role") == "prompter":
                cur_q = m.get("text", "")
            elif m.get("role") == "assistant" and cur_q:
                if (m.get("rank") or 0) <= 1 and is_clean_fr(cur_q) and is_clean_fr(m.get("text", "")):
                    rows.append({"user": cur_q.strip()[:2000], "guizmo": m.get("text", "").strip()[:2000]})
                cur_q = None
    print(f"OASST FR : {len(rows)} paires")
    return rows


def save_lines(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in lines:
            one = " ".join(t.split())
            if one:
                f.write(one + "\n")
