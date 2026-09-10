"""SFT Coeur : Vie 60% / Raison 30% / Chercheur 10% (mix impose par README).
Usage: python -m scripts.train_sft --tokenizer tokenizer --vie data/seed/vie.jsonl --raison data/seed/raison.jsonl --chercheur data/seed/chercheur.jsonl --from_ckpt checkpoints/.../final.pt --tiny
"""
import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from guizmo.config import GuizmoConfig
from guizmo.model import GuizmoForCausalLM
from guizmo.tokenizer import GuizmoTokenizer
from guizmo.data import load_jsonl, mix_datasets, SFTDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer")
    ap.add_argument("--vie", default="data/seed/vie.jsonl")
    ap.add_argument("--raison", default="data/seed/raison.jsonl")
    ap.add_argument("--chercheur", default="data/seed/chercheur.jsonl")
    ap.add_argument("--from_ckpt", default=None)
    ap.add_argument("--out", default="checkpoints/guizmo-150m-coeur")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = GuizmoConfig.tiny() if args.tiny else GuizmoConfig()
    tok = GuizmoTokenizer(f"{args.tokenizer}/vocab.json", f"{args.tokenizer}/merges.txt")
    cfg.vocab_size = len(tok.vocab)
    vie = load_jsonl(args.vie) if Path(args.vie).exists() else []
    raison = load_jsonl(args.raison) if Path(args.raison).exists() else []
    cher = load_jsonl(args.chercheur) if Path(args.chercheur).exists() else []
    mixed = mix_datasets(vie, raison, cher)
    assert mixed, "datasets seed vides -> lance scripts/build_datasets.py"
    print(f"mix SFT: total={len(mixed)} (vie={len(vie)} raison={len(raison)} chercheur={len(cher)})")

    ds = SFTDataset(mixed, tok, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    model = GuizmoForCausalLM(cfg).to(args.device)
    if args.from_ckpt:
        from guizmo.model import load_ckpt_compatible
        load_ckpt_compatible(model, args.from_ckpt, args.device)
        print(f"resume depuis {args.from_ckpt}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    model.train()
    for ep in range(args.epochs):
        tot = 0.0
        for b in dl:
            x = b["input_ids"].to(args.device)
            y = b["labels"].to(args.device)
            opt.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        print(f"epoch {ep+1}/{args.epochs} loss={tot/max(1,len(dl)):.3f}")
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, f"{args.out}/coeur-ep{ep+1}.pt")
    print(f"OK -> {args.out}")


if __name__ == "__main__":
    main()
