"""Dataset ROUTEUR : le nano apprend intention -> requetes, pas le monde.
Format : <user> question <route> intent | q1 ; q2 </route>
Genere depuis tes 780 dialogues + cas limites (suivi, salutation, emotion).
"""
import json
from pathlib import Path


def main():
    from scripts.gen_seed import gen_vie, gen_raison, gen_chercheur
    from guizmo.router import route
    rows = gen_vie(500) + gen_raison(200) + gen_chercheur(80)
    extra = [
        {"user": "Salut", "guizmo": ""}, {"user": "Wesh ça va ?", "guizmo": ""},
        {"user": "Je suis triste ce soir", "guizmo": ""},
        {"user": "Et lui il a gagné quoi ?", "guizmo": ""},
        {"user": "Pourquoi pas ?", "guizmo": ""},
        {"user": "iPhone ou Samsung ?", "guizmo": ""},
        {"user": "C'est quoi le prix du bitcoin aujourd'hui ?", "guizmo": ""},
    ]
    hist = [{"user": "Qui a gagné la coupe du monde 2022 ?"}]
    out = []
    for r in rows + extra:
        u = r["user"]
        h = hist if u in ("Et lui il a gagné quoi ?", "Pourquoi pas ?") else None
        rt = route(u, h)
        q = " ; ".join(rt.queries)
        out.append({"user": u, "route": f"{rt.intent} | {q}"})
    p = Path("data/seed/routeur.jsonl")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"routeur: {len(out)} exemples -> {p}")
    print(Counter(r["route"].split(" | ")[0] for r in out))


if __name__ == "__main__":
    main()
