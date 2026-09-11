"""Flux V2 chercheur systematique : route -> search x1-3 -> synthese.
Le nano ne stocke rien : il comprend, cherche a chaque fois, synthetise.
"""
from .router import route
from .search import web_search, format_for_coeur
from .synth import synth_prompt, fallback_answer


def answer(user, history=None, gen_fn=None, max_results=4):
    rt = route(user, history)
    if not rt.needs_search:
        if gen_fn:
            return gen_fn(synth_prompt(user, rt, "")), rt
        return fallback_answer(user, rt), rt
    blocks, n = [], 0
    for q in rt.queries[:3]:
        try:
            res = web_search(q, max_results=max_results)
            snips = [s for s in res.snippets if "indisponible" not in s
                     and "aucun resultat" not in s]
            n += len(snips)
            if snips:
                blocks.append(format_for_coeur(res))
        except Exception:
            continue
    block = "\n".join(blocks)
    if gen_fn:
        if not block:
            return fallback_answer(user, rt, 0), rt
        return gen_fn(synth_prompt(user, rt, block)), rt
    if not block:
        return fallback_answer(user, rt, 0), rt
    # sans modele : synthese extractive honnete (jamais d'invention)
    return extractive(user, rt, blocks), rt


def extractive(user, rt, blocks):
    import re
    bullets = re.findall(r"^- (.+)$", blocks, re.M)
    keep = [b.strip()[:220] for b in bullets if len(b.strip()) > 40][:4]
    if not keep:
        return fallback_answer(user, rt, 0)
    txt = "Voilà ce que je trouve : " + " ".join(keep[:3])
    tail = {"avis": "Si tu veux mon avis : dis-moi ton usage exact et je tranche.",
            "comparaison": "Dis-moi ton budget/usage et je tranche entre les deux.",
            "actu": "Tu veux que je surveille le sujet et que je te tienne au jus ?",
            "tuto": "Tu veux qu'on le fasse étape par étape ensemble ?",
            "definition": "Tu veux un exemple concret pour fixer ça ?",
            "factuel": "C'est pour quoi exactement, je peux creuser un angle ?",
            "suivi": "Je continue à creuser ou on passe à la suite ?"}.get(
                rt.intent, "T'en penses quoi, on creuse ?")
    return txt + " " + tail
