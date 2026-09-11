"""Demo web Guizmo V3 : route -> web -> synthese (zero hallucination, zero GPU).
Lancement : python app.py  (puis ouvrir http://127.0.0.1:7860)
Pas de modele a charger : le routeur a regles + recherche DDG/Wikipedia + synthese
extractive. Le savoir vient du web, jamais d'invention. Voir DEMARRAGE.md §0.
"""
import gradio as gr
from guizmo.pipeline import answer


def _hist_dicts(hist):
    """Gradio donne des paires (ou dicts) -> format {'user','guizmo'} du routeur."""
    out = []
    for m in (hist or []):
        if isinstance(m, dict):
            out.append({"user": m.get("user", ""), "guizmo": m.get("guizmo", "")})
        elif isinstance(m, (list, tuple)) and len(m) >= 2:
            out.append({"user": m[0], "guizmo": m[1] or ""})
    return out


def chat_fn(msg, hist):
    try:
        txt, rt = answer(msg, _hist_dicts(hist))
        tag = "\n\n_[{} | search={} | q={}]_".format(
            rt.intent, rt.needs_search, "; ".join(rt.queries[:2]) or "-")
        return txt + tag
    except Exception as e:
        return (f"(oops, j'ai perdu le fil : {e}) — reformule avec un mot-cle "
                f"en plus (ville, annee, modele) et je re-check direct.")


with gr.Blocks(title="Guizmo V3") as demo:
    gr.Markdown(
        "# Guizmo — le petit francais qui ne sait rien, mais comprend tout\n"
        "Route -> recherche web (DuckDuckGo + Wikipedia FR) -> synthese avec avis.\n"
        "Zero hallucination : il ne dit que ce que le web a dit, avec son ton.")
    gr.ChatInterface(chat_fn)

if __name__ == "__main__":
    demo.launch()
