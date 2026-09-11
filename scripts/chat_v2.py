"""Inference CLI V3 : route (regles OU nano) -> search x1-3 -> synthese.
Sans ckpt : regles + pipeline extractive (zero hallucination).
Avec --routeur : le nano genere <route>intent | q1 ; q2</route>, on le parse
et le MEME pipeline fait search + synthese. Si le nano sort du sale, fallback
total sur le routeur a regles (garde-fou, jamais de crash).
"""
import argparse
import torch
from guizmo.config import GuizmoConfig
from guizmo.model import GuizmoForCausalLM
from guizmo.tokenizer import GuizmoTokenizer
from guizmo.pipeline import answer_with_route
from guizmo.router import route as rule_route, Route, INTENTS


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


def parse_route(raw, user):
    """'<route>intent | q1 ; q2</route>' -> Route fiable (fallback regles si sale)."""
    raw = raw.replace("</route>", "").strip()
    intent, queries = None, []
    if "|" in raw:
        left, right = raw.split("|", 1)
        cand = left.strip().split()[0] if left.strip() else ""
        if cand in INTENTS:
            intent = cand
        queries = [q.strip() for q in right.split(";") if q.strip()][:3]
    else:
        queries = [raw.strip()[:120]] if raw.strip() else []
    if intent is None:
        return rule_route(user)  # garde-fou : nano a rate le format
    if not queries:
        queries = rule_route(user).queries
    return Route(intent=intent, needs_search=bool(queries), queries=queries,
                 context_summary="", confidence=0.6, reason="route nano")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer")
    ap.add_argument("--routeur", default=None, help="ckpt nano routeur (optionnel)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_results", type=int, default=4)
    args = ap.parse_args()

    model = tok = None
    if args.routeur:
        model, tok = load_nano(args.routeur, args.tokenizer, args.device)
        print(f"nano routeur charge ({model.num_params()/1e6:.1f}M).")
    else:
        print("Guizmo V3 (regles + web, zero hallucination). 'quit' pour sortir.")
    hist = []
    while True:
        u = input("\nToi : ").strip()
        if u.lower() in ("quit", "exit"):
            break
        if model is not None:
            prompt = f"<user> {u} <route> "
            ids = torch.tensor([tok.encode(prompt, add_bos=True, add_eos=False)],
                               dtype=torch.long).to(args.device)
            out = model.generate(ids, max_new_tokens=64, temperature=0.3, top_k=20)
            gen = tok.decode(out[0].tolist())
            raw = gen[len(tok.decode(ids[0].tolist())):]
            rt = parse_route(raw, u)
            print(f"  [route nano: {rt.intent} | {rt.queries}]")
        else:
            rt = rule_route(u)
        txt, rt = answer_with_route(u, rt, max_results=args.max_results)
        print(f"  [{rt.intent} | search={rt.needs_search} | q={rt.queries}]")
        print("Guizmo :", txt)
        hist += [{"user": u, "guizmo": txt}]


if __name__ == "__main__":
    main()
