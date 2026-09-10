"""Inference CLI : python -m scripts.chat --ckpt checkpoints/... --tokenizer tokenizer/"""
import argparse
import torch
from guizmo.config import GuizmoConfig
from guizmo.model import GuizmoForCausalLM
from guizmo.tokenizer import GuizmoTokenizer
from guizmo.converse import needs_search, build_augmented_prompt
from guizmo.search import web_search, format_for_coeur


def load_all(ckpt: str, tok_dir: str, device: str):
    tok = GuizmoTokenizer(f"{tok_dir}/vocab.json", f"{tok_dir}/merges.txt")
    sd = torch.load(ckpt, map_location=device)
    saved = sd.get("cfg", {}) if isinstance(sd, dict) else {}
    cfg = GuizmoConfig()
    for k, v in saved.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.vocab_size = len(tok.vocab)  # aligne embeddings sur tokenizer reel
    model = GuizmoForCausalLM(cfg)
    model.load_state_dict_resize(sd.get("model", sd))
    model.to(device).eval()
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="tokenizer")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--system", default="Tu es Guizmo, le petit francais qui ne sait rien mais comprend tout. Empathique, drole, avec des avis.")
    args = ap.parse_args()
    model, tok = load_all(args.ckpt, args.tokenizer, args.device)
    print("Guizmo pret. Tape 'quit' pour sortir.")
    while True:
        u = input("\nToi : ").strip()
        if u.lower() in ("quit", "exit"):
            break
        q = needs_search(u)
        aug = ""
        if q:
            print(f"  [Guizmo check : <search>{q}</search>]")
            aug = format_for_coeur(web_search(q))
        prompt = build_augmented_prompt(u, aug, args.system) if aug else f"{args.system}\n<user> {u} <guizmo> "
        ids = torch.tensor([tok.encode(prompt, add_bos=True, add_eos=False)], dtype=torch.long)
        out = model.generate(ids.to(args.device), max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        txt = tok.decode(out[0].tolist())
        # affiche seulement la fin generee
        print("Guizmo :", txt[len(tok.decode(ids[0].tolist())):])


if __name__ == "__main__":
    main()
