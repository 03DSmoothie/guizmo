"""Datasets de Guizmo : Vie 60% / Raison 30% / Chercheur 10% (cf. README).
+ utilitaires PyTorch pour pretrain (texte brut FR) et SFT (dialogues).
"""
import json
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset


def load_jsonl(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def mix_datasets(vie: list[dict], raison: list[dict], chercheur: list[dict], seed: int = 42) -> list[dict]:
    """Melange stratifie : 60/30/10. Chaque item a deja user/guizmo (+search eventuel)."""
    rnd = random.Random(seed)
    vie = list(vie); raison = list(raison); chercheur = list(chercheur)
    rnd.shuffle(vie); rnd.shuffle(raison); rnd.shuffle(chercheur)
    # proportions cibles
    total = len(vie) + len(raison) + len(chercheur)
    if total == 0:
        return []
    # on sur/sous-echantillonne pour respecter 60/30/10 sur la taille du plus grand melange possible
    n = max(len(vie + raison + chercheur), 100)
    n_vie, n_raison = int(n * 0.6), int(n * 0.3)
    n_cher = n - n_vie - n_raison
    def take(pool, k):
        if not pool:
            return []
        return [pool[i % len(pool)] for i in range(k)]
    mixed = take(vie, n_vie) + take(raison, n_raison) + take(chercheur, n_cher)
    rnd.shuffle(mixed)
    return mixed


class SFTDataset(Dataset):
    """Encode <user>...<guizmo>... ; loss masquee (-100) sur le prompt user."""

    def __init__(self, rows: list[dict], tokenizer, max_len: int = 512):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        prompt = f"<user> {r['user']} <guizmo> "
        full = prompt + r["guizmo"]
        p_ids = self.tok.encode(prompt, add_bos=True, add_eos=False)
        f_ids = self.tok.encode(full, add_bos=True, add_eos=True)[: self.max_len]
        labels = list(f_ids)
        for j in range(min(len(p_ids), len(labels))):
            labels[j] = -100
        # pad
        pad = self.tok.pad_id
        L = len(f_ids)
        ids = f_ids + [pad] * (self.max_len - L)
        labs = labels + [-100] * (self.max_len - L)
        attn = [1] * L + [0] * (self.max_len - L)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labs, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


class PretrainDataset(Dataset):
    """LM causal sur texte brut tokenize en blocs de block_size."""

    def __init__(self, texts: list[str], tokenizer, block_size: int = 512):
        ids: list[int] = []
        for t in texts:
            ids += tokenizer.encode(t, add_bos=False, add_eos=True)
        # decoupe en blocs
        self.blocks = [ids[i : i + block_size] for i in range(0, len(ids) - block_size + 1, block_size)]
        self.block_size = block_size

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, i):
        b = self.blocks[i]
        x = torch.tensor(b, dtype=torch.long)
        return {"input_ids": x, "labels": x.clone()}
