"""Inference CLI V2 : route -> search x1-3 -> synthese (extractive ou nano).
Sans ckpt : pipeline extractive (zero hallucination). Avec --routeur :
le nano genere les requetes, le pipeline garde le garde-fou regles.
"""
import argparse
import torch
from guizmo.config import GuizmoConfig
from guizmo.model import GuizmoForCausalLM
from guizmo.tokenizer import GuizmoTokenizer
from guizmo.pipeline import answer as pipe_answer
from guizmo.router import route as rule_route


def load_nano(ckpt, tok_dir, device):
    tok = GuizmoTokenizer(f"{tok_dir}/vocab.json", f"{tok_dir}/merges.txt")
    sd = torch.load(ckpt, map_location=device)
    saved = sd.get("cfg", {}) if isinstance(sd, dict) else {}
    cfg = GuizmoConfig.nano()
    for k, v in saved.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.vocab_size = len(tok.vocab)
    m = GuizmoForCausalLM(cfg)
    m.load_state_dict_resize(sd.get("model", sd))
    return m.to(device).eval(), tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer")
    ap.add_argument("--routeur", default=None, help="ckpt nano routeur (optionnel)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_results", type=int, default=4)
    args = ap.parse_args()

    gen = None
    if args.routeur:
        model, tok = load_nano(args.routeur, args.tokenizer, args.device)
        print(f"nano routeur charge ({model.num_params()/1e6:.1f}M).")
    else:
        print("Guizmo V2 (regles + web, zero hallucination). 'quit' pour sortir.")
        tok = None
    hist = []
    while True:
        u = input("\nToi : ").strip()
        if u.lower() in ("quit", "exit"):
            break
        if args.routeur:
            prompt = f"<user> {u} <route> "
            ids = torch.tensor([tok.encode(prompt, add_bos=True, add_eos=False)],
                               dtype=torch.long).to(args.device)
            out = model.generate(ids, max_new_tokens=64, temperature=0.3, top_k=20)
            raw = tok.decode(out[0].tolist())[len(tok.decode(ids[0].tolist())):]
            print(f"  [route nano: {raw.strip()[:120]}]")
            txt, rt = pipe_answer(u, hist)
        else:
            txt, rt = pipe_answer(u, hist)
            print(f"  [{rt.intent} | search={rt.needs_search} | q={rt.queries}]")
        print("Guizmo :", txt)
        hist += [{"user": u, "guizmo": txt}]


if __name__ == "__main__":
    main()
