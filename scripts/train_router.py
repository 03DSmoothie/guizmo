"""Train ROUTEUR nano (~10M) : intention -> requetes. CPU OK, Kaggle sans crash.
Usage: python -m scripts.train_router --tokenizer tokenizer --nano --epochs 5
Le nano apprend le FORMAT <route>intent | q1 ; q2</route>, pas des faits.
"""
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from guizmo.config import GuizmoConfig
from guizmo.model import GuizmoForCausalLM
from guizmo.tokenizer import GuizmoTokenizer
from guizmo.data import load_jsonl


class RouterDS(Dataset):
    def __init__(self, rows, tok, max_len=128):
        self.rows, self.tok, self.max_len = rows, tok, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        full = f"<user> {r['user']} <route> {r['route']}"
        ids = self.tok.encode(full, add_bos=True, add_eos=True)[:self.max_len]
        pad = self.tok.pad_id
        L = len(ids)
        ids = ids + [pad] * (self.max_len - L)
        attn = [1] * L + [0] * (self.max_len - L)
        x = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": x, "labels": x.clone(),
                "attention_mask": torch.tensor(attn, dtype=torch.long)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer")
    ap.add_argument("--data", default="data/seed/routeur.jsonl")
    ap.add_argument("--out", default="checkpoints/guizmo-nano-routeur")
    ap.add_argument("--nano", action="store_true", help="nano ~10M (defaut)")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.tiny:
        cfg = GuizmoConfig.tiny()
    else:
        cfg = GuizmoConfig.nano()
    tok = GuizmoTokenizer(f"{args.tokenizer}/vocab.json", f"{args.tokenizer}/merges.txt")
    cfg.vocab_size = len(tok.vocab)
    rows = load_jsonl(args.data) if Path(args.data).exists() else []
    assert rows, "lance d'abord python -m scripts.build_router"
    ds = RouterDS(rows, tok, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    model = GuizmoForCausalLM(cfg).to(args.device)
    print(f"Params routeur: {model.num_params()/1e6:.1f}M (vocab={len(tok.vocab)})")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.train()
    for ep in range(args.epochs):
        tot = 0.0
        for b in dl:
            opt.zero_grad()
            _, loss = model(b["input_ids"].to(args.device), b["labels"].to(args.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        print(f"epoch {ep+1}/{args.epochs} loss={tot/max(1,len(dl)):.3f}")
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__},
                   f"{args.out}/routeur-ep{ep+1}.pt")
    print(f"OK -> {args.out}")


if __name__ == "__main__":
    main()
