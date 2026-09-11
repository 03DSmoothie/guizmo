"""Routeur d'intention V2 : leger, sans connaissances.
Comprend INTENTION + CONTEXTE + TYPE, puis decide : chercher ? quelles requetes ?
Le nano (~10M) apprend ca, pas le monde. Le monde vient du web.
"""
import re
from dataclasses import dataclass, field

INTENTS = ("salutation", "emotion", "avis", "definition", "factuel",
           "comparaison", "tuto", "actu", "suivi")

SALUT_RE = re.compile(r"^(salut|hello|bonjour|bonsoir|yo|wesh|coucou|bye)\b", re.I)
SUIVI_RE = re.compile(r"^(et\s+(lui|elle|eux)|pourquoi(\s+pas)?\s*\??$|et pourquoi|continue)", re.I)
DEF_RE = re.compile(r"(c'est quoi|qu'est-ce que|veut dire quoi|explique\w*\s+(moi\s+)?(pourquoi|comment|ce qu))", re.I)

EMOTION_W = ("triste", "stress", "angoiss", "peur", "marre", "saoule", "ennuie",
             "seul", "pleure", "colere", "fatigu", "motive")
AVIS_W = ("t'en penses", "ton avis", "tu preferes", "t'aimes", "conseil",
          "tu ferais quoi", "je fais quoi", "dois-je")
COMPA_W = (" vs ", " ou ", "compar", "difference", "mieux", "lequel", "laquelle")
TUTO_W = ("comment", "tuto", "recette", "apprendre", "etape", "faire pour")
ACTU_W = ("dernier", "derniere", "2024", "2025", "2026", "aujourd", "hier",
          "demain", "actu", "news", "score", "resultat", "prix", "meteo",
          "ballon d'or", "election", "sortie", "transfert")


@dataclass
class Route:
    intent: str
    needs_search: bool
    queries: list[str] = field(default_factory=list)
    context_summary: str = ""
    confidence: float = 0.75
    reason: str = ""



def _clean(q):
    return " ".join(q.strip().split())[:300]


def detect_intent(text, history=None):
    t = text.strip()
    low = t.lower()
    if SUIVI_RE.search(low) and history:
        return "suivi"
    if SALUT_RE.search(low) and len(t.split()) <= 6:
        return "salutation"
    if any(w in low for w in ACTU_W):
        return "actu"
    if DEF_RE.search(low) or low.startswith(("pourquoi", "comment se fait")):
        return "definition"
    if any(w in low for w in COMPA_W) and "?" in t:
        return "comparaison"
    if any(w in low for w in TUTO_W):
        return "tuto"
    if any(w in low for w in AVIS_W) or low.startswith("tu "):
        return "avis"
    if any(w in low for w in EMOTION_W) or low.startswith(
            ("j'aime pas", "je suis ", "je me sens")):
        if "pourquoi" in low and ("tort" in low or "raison" in low):
            return "avis"
        return "emotion"
    if "?" in t:
        return "factuel"
    return "avis"


def _queries_for(intent, text, history):
    t = _clean(text)
    if intent == "suivi" and history:
        for m in reversed(history[-4:]):
            u = m.get("user", "") if isinstance(m, dict) else ""
            if u and len(u.split()) > 3:
                t = f"{_clean(u)} {t}"
                break
    base = re.sub(r"^(wesh guizmo,?|hey,?|salut,?|stp,?)\s*", "", t, flags=re.I)
    base = re.sub(r"\s*\?$", "", base).strip()
    if intent == "salutation":
        return []
    if intent == "emotion" and "?" not in text:
        return []
    # V2 chercheur systematique : TOUT le reste passe au web (le nano ne sait rien)
    qs = [base]
    low = base.lower()
    if intent == "comparaison" and " ou " in low:
        parts = re.split(r"\s+ou\s+", base, maxsplit=1)
        if len(parts) == 2:
            qs = [parts[0].strip(), parts[1].strip(), base]
    elif intent == "definition":
        qs = [base, base + " explication simple"]
    elif intent in ("actu", "factuel"):
        qs = [base, base + " 2025 2026"]
    elif intent == "tuto":
        qs = [base + " tuto", base + " etapes"]
    elif intent == "avis":
        core = re.sub(r"(t'en penses quoi.*|ton avis.*|je fais quoi.*)\??$",
                      "", base, flags=re.I).strip()
        qs = [core or base, (core or base) + " avis comparatif"] if core else [base]
    seen, out = set(), []
    for q in qs:
        q = _clean(q)
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
        if len(out) == 3:
            break
    return out


WHY = {"salutation": "social, reponds direct",
       "emotion": "emotion pure -> ecoute, pas de search",
       "avis": "cherche des faits pour nourrir ton avis, puis tranche",
       "definition": "cadrage web puis explique simple",
       "factuel": "fait verifiable -> search systematique",
       "comparaison": "1 search par option + 1 globale",
       "tuto": "search puis plan d-action",
       "actu": "frais -> search 2025/2026 obligatoire",
       "suivi": "reprend le sujet precedent"}


def route(text, history=None):
    intent = detect_intent(text, history)
    queries = _queries_for(intent, text, history)
    needs = bool(queries)
    ctx = ""
    if history:
        last = [m.get("user", "")[:80] for m in history[-3:]
                if isinstance(m, dict) and m.get("user")]
        ctx = " | ".join(last)[-240:]
    conf = 0.9 if intent in ("salutation", "actu") else 0.75
    why = WHY.get(intent, "")
    if needs:
        why = "V2 : tu ne sais rien, tu cherches. " + why
    return Route(intent=intent, needs_search=needs, queries=queries,
                 context_summary=ctx, confidence=conf, reason=why)
