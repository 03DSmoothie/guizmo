"""Pretrain Coeur Guizmo : LM causal sur texte FR brut. 0 EUR -> CPU/Colab/Kaggle.
Usage: python -m scripts.train_pretrain --tokenizer tokenizer --train data/processed/pretrain.txt --tiny
"""
import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from guizmo.config import GuizmoConfig
from guizmo.model import GuizmoForCausalLM
from guizmo.tokenizer import GuizmoTokenizer
from guizmo.data import PretrainDataset


def cosine_sched(step, warmup, total, lr, min_lr):
    if step < warmup:
        return lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + __import__("math").cos(3.14159 * min(p, 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer")
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", default="checkpoints/guizmo-150m-pretrain")
    ap.add_argument("--tiny", action="store_true", help="modele 30M pour tester gratuit")
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = GuizmoConfig.tiny() if args.tiny else GuizmoConfig()
    tok = GuizmoTokenizer(f"{args.tokenizer}/vocab.json", f"{args.tokenizer}/merges.txt")
    cfg.vocab_size = len(tok.vocab)
    print(f"Params: {cfg.count_params()/1e6:.1f}M (vocab={cfg.vocab_size})")
    texts = Path(args.train).read_text(encoding="utf-8").splitlines()
    texts = [t for t in texts if t.strip()]
    ds = PretrainDataset(texts, tok, args.block_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model = GuizmoForCausalLM(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    Path(args.out).mkdir(parents=True, exist_ok=True)

    model.train()
    step = 0
    it = iter(dl)
    import tqdm
    pbar = tqdm.tqdm(total=args.max_steps, desc="pretrain guizmo")
    while step < args.max_steps:
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl)
            b = next(it)
        x = b["input_ids"].to(args.device)
        lr_now = cosine_sched(step, 100, args.max_steps, args.lr, args.lr / 10)
        for pg in opt.param_groups:
            pg["lr"] = lr_now
        opt.zero_grad()
        _, loss = model(x, x)
        (loss / args.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{lr_now:.2e}")
        if step % 500 == 0:
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, f"{args.out}/step{step}.pt")
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, f"{args.out}/final.pt")
    print(f"OK -> {args.out}/final.pt")


if __name__ == "__main__":
    main()
