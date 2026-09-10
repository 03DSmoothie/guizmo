"""Demo web Gradio : Coeur + Yeux. Lancement: python app.py"""
import torch
import gradio as gr
from guizmo.config import GuizmoConfig
from guizmo.model import GuizmoForCausalLM
from guizmo.tokenizer import GuizmoTokenizer
from guizmo.converse import needs_search, build_augmented_prompt
from guizmo.search import web_search, format_for_coeur

MODEL = None
TOK = None
CKPT = "checkpoints/guizmo-150m-coeur/coeur-ep3.pt"
TOKDIR = "tokenizer"
SYSTEM = "Tu es Guizmo, le petit francais qui ne sait rien mais comprend tout. Empathique, drole, avec des avis."


def lazy_load():
    global MODEL, TOK
    if MODEL is None:
        try:
            TOK = GuizmoTokenizer(f"{TOKDIR}/vocab.json", f"{TOKDIR}/merges.txt")
            import torch as _t
            sd = _t.load(CKPT, map_location="cpu")
            saved = sd.get("cfg", {}) if isinstance(sd, dict) else {}
            cfg = GuizmoConfig()
            for k, v in saved.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            cfg.vocab_size = len(TOK.vocab)
            MODEL = GuizmoForCausalLM(cfg)
            MODEL.load_state_dict_resize(sd.get("model", sd))
            MODEL.eval()
        except Exception as e:
            return f"(mode demo sans poids : {e})"
    return ""


def chat_fn(msg, hist):
    warn = lazy_load()
    if MODEL is None:
        return warn + f"\nGuizmo (demo) : Je comprends ! '{msg}' — entraine-moi (voir README DEMARRAGE) et je te repondrai avec mon Coeur."
    q = needs_search(msg)
    aug = format_for_coeur(web_search(q)) if q else ""
    prompt = build_augmented_prompt(msg, aug, SYSTEM) if aug else f"{SYSTEM}\n<user> {msg} <guizmo> "
    ids = torch.tensor([TOK.encode(prompt, add_bos=True, add_eos=False)], dtype=torch.long)
    out = MODEL.generate(ids, max_new_tokens=200, temperature=0.8)
    txt = TOK.decode(out[0].tolist())
    return txt[len(TOK.decode(ids[0].tolist())):]


with gr.Blocks(title="Guizmo") as demo:
    gr.Markdown("# Guizmo — le petit francais qui pense avant de chercher")
    gr.ChatInterface(chat_fn)

if __name__ == "__main__":
    demo.launch()
