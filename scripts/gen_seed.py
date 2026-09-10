"""Generateur de sang : 500 Vie + 200 Raison avec TON ton (2h -> 30 sec).
Chaque sujet -> 5 formulations user differentes x reponse Guizmo.
Verifie : python -m scripts.gen_seed
"""
import random
from scripts.seed_ton import OUVERTURES, EMPATHIE, AVIS, RELANCES, CLOSINGS
from scripts.seed_vie_topics_a import TOPICS_A
from scripts.seed_vie_topics_b import TOPICS_B
from scripts.seed_vie_topics_c import TOPICS_C
from scripts.seed_vie_topics_d import TOPICS_D
from scripts.seed_raison_topics_a import RAISON_A
from scripts.seed_raison_topics_b import RAISON_B
from scripts.seed_raison_topics_c import RAISON_C
from scripts.seed_raison_topics_d import RAISON_D
from scripts.seed_raison_e import RAISON_E

USER_PATTERNS = [
    "Wesh Guizmo, {sujet} : {avis_court} non ?",
    "J'aime pas {sujet} parce que {pb}. Explique moi pourquoi j'ai tort.",
    "Franchement {sujet} ca me saoule. T'en penses quoi toi ?",
    "Hey, parle moi de {sujet}. J'ai besoin de ton avis honnete.",
    "{sujet_cap} : je sais pas quoi en penser. Aide moi.",
]

POLEMIQUE_PB = ["c'est trop long", "ca me parle pas", "j'ai deja essaye et bof", "tout le monde en parle trop", "ca me stresse"]


def gen_vie(n_total=500, seed=7):
    rnd = random.Random(seed)
    base = TOPICS_A + TOPICS_B + TOPICS_C + TOPICS_D  # 100 sujets
    rows = []
    i = 0
    while len(rows) < n_total:
        sujet, avis = base[i % len(base)]
        pat = USER_PATTERNS[(i // len(base)) % len(USER_PATTERNS)]
        pb = rnd.choice(POLEMIQUE_PB)
        mots = sujet.split()
        cap = sujet[0].upper() + sujet[1:] if sujet else sujet
        user = pat.format(sujet=sujet, sujet_cap=cap, avis_court=avis.split(",")[0][:60], pb=pb)
        ouv = rnd.choice(OUVERTURES)
        emp = rnd.choice(EMPATHIE)
        av = rnd.choice(AVIS)
        rel = rnd.choice(RELANCES)
        guizmo = f"{ouv} {emp}, {avis}. {av} : dis-moi, {rel} {rnd.choice(CLOSINGS)}"
        # varie la longueur : 1 fois sur 3 reponse courte punchy
        if i % 3 == 2:
            guizmo = f"{ouv} {avis}. {rel}"
        rows.append({"user": user.strip(), "guizmo": guizmo.strip()})
        i += 1
    return rows[:n_total]


RAISON_PATTERNS = [
    "Explique moi {q} comme a un pote.",
    "Pourquoi {sujet} ? Vas-y simple stp.",
    "Wesh Guizmo, {q} ? J'ai jamais compris.",
    "{q_cap} ? Explique avec un exemple concret.",
]


def gen_raison(n_total=200, seed=11):
    rnd = random.Random(seed)
    base = RAISON_A + RAISON_B + RAISON_C + RAISON_D + RAISON_E  # 100 sujets
    rows = []
    i = 0
    while len(rows) < n_total:
        q, expli = base[i % len(base)]
        pat = RAISON_PATTERNS[(i // len(base)) % len(RAISON_PATTERNS)]
        sujet = q.replace("pourquoi ", "").strip()
        cap = q[0].upper() + q[1:] if q else q
        user = pat.format(q=q, q_cap=cap, sujet=sujet)
        ouv = rnd.choice(["Alors imagine", "Ok, simple", "Je t'explique comme a un pote", "Easy"])
        rel = rnd.choice(RELANCES)
        guizmo = f"{ouv} : {expli}. {rel}"
        if i % 4 == 3:
            guizmo = f"{ouv} : {expli}."
        rows.append({"user": user.strip(), "guizmo": guizmo.strip()})
        i += 1
    return rows[:n_total]


VIE_EXTRA_CHERCHEUR = [
    {"user": "C'est qui le dernier ballon d'or ?", "guizmo": "Attends, je ne l'ai plus en tete, je check. <search>ballon d'or vainqueur 2024 2025</search>"},
    {"user": "Il fait quel temps a Paris demain ?", "guizmo": "La meteo change trop vite, je verifie. <search>meteo Paris demain</search>"},
    {"user": "C'est quoi le score du PSG hier ?", "guizmo": "Je veux pas te dire une betise, je check le score. <search>score PSG match hier</search>"},
    {"user": "Combien vaut le bitcoin aujourd'hui ?", "guizmo": "Ca bouge toutes les minutes, je regarde. <search>prix bitcoin aujourd'hui euro</search>"},
    {"user": "C'est quoi le dernier film Marvel sorti ?", "guizmo": "Je check vite fait pour pas te sortir un vieux titre. <search>dernier film Marvel sorti 2025 2026</search>"},
    {"user": "Qui a gagne les dernieres elections ?", "guizmo": "Attends, je verifie pour etre sur. <search>dernieres elections resultat vainqueur</search>"},
    {"user": "Il sort quand le prochain iPhone ?", "guizmo": "Je check les dernieres annonces. <search>prochain iPhone date sortie prix</search>"},
    {"user": "C'est quoi la nouvelle saison de Stranger Things ?", "guizmo": "Je verifie vite fait. <search>Stranger Things nouvelle saison date sortie</search>"},
]


def gen_chercheur(n_total=80, seed=13):
    rnd = random.Random(seed)
    rows = []
    i = 0
    queries = ["meteo Lyon demain", "prix essence aujourd'hui", "resultat loto dernier tirage",
               "concert Paris ce weekend", "sortie film cinema cette semaine", "transfert foot mercato rumeur"]
    while len(rows) < n_total:
        if i < len(VIE_EXTRA_CHERCHEUR):
            rows.append(VIE_EXTRA_CHERCHEUR[i])
        else:
            q = rnd.choice(queries)
            rows.append({"user": f"Wesh, tu sais pour {q} ?",
                         "guizmo": f"Je veux pas te dire de betise, je check. <search>{q}</search>"})
        i += 1
    return rows[:n_total]


if __name__ == "__main__":
    v = gen_vie(500)
    r = gen_raison(200)
    c = gen_chercheur(80)
    print(f"VIE={len(v)} RAISON={len(r)} CHERCHEUR={len(c)}")
    print("ex Vie:", v[0])
    print("ex Raison:", r[0])
