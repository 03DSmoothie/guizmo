"""Eval rapide du Coeur : perplexite + check personnalite (pas de recrachage)."""
import argparse
import math
import torch
from guizmo.config import GuizmoConfig
from guizmo.model import GuizmoForCausalLM
from guizmo.tokenizer import GuizmoTokenizer
from guizmo.data import load_jsonl


def perplexity(model, tok, texts, device, max_len=256):
    model.eval()
    nll, ntok = 0.0, 0
    with torch.no_grad():
        for t in texts:
            ids = tok.encode(t)[:max_len]
            x = torch.tensor([ids], dtype=torch.long).to(device)
            logits, _ = model(x, x)
            import torch.nn.functional as F
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1), reduction="sum")
            nll += loss.item()
            ntok += len(ids)
    return math.exp(nll / max(1, ntok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="tokenizer")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    tok = GuizmoTokenizer(f"{args.tokenizer}/vocab.json", f"{args.tokenizer}/merges.txt")
    sd = torch.load(args.ckpt, map_location=args.device)
    saved = sd.get("cfg", {}) if isinstance(sd, dict) else {}
    cfg = GuizmoConfig()
    for k, v in saved.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.vocab_size = len(tok.vocab)
    model = GuizmoForCausalLM(cfg)
    from guizmo.model import load_ckpt_compatible
    load_ckpt_compatible(model, args.ckpt, args.device)
    model.to(args.device)
    texts = [r["user"] + " " + r["guizmo"] for r in load_jsonl("data/seed/vie.jsonl")]
    print(f"perplexite seed vie: {perplexity(model, tok, texts, args.device):.2f}")
    print("check ton Guizmo : il dit 'je comprends', 'moi aussi', donne un avis, pas 'Selon Allocine'.")


if __name__ == "__main__":
    main()
