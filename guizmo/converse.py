"""Flux final Guizmo : pense -> cherche si besoin -> digere avec son Coeur.
Logique imposee par le README : ne recrache jamais internet, re-ecrit avec avis/ton.
"""
import re
from .search import web_search, format_for_coeur

SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)


def needs_search(text: str) -> str | None:
    """Heuristique + balise explicite <search>. Retourne la requete ou None."""
    m = SEARCH_RE.search(text)
    if m:
        return m.group(1).strip()
    triggers = ["qui est", "c'est qui", "dernier", "derniere", "2024", "2025", "2026",
                "combien", "quand", "ou se", "où se", "prix", "score", "vainqueur",
                "ballon d'or", "meteo", "météo", "news", "actu"]
    low = text.lower()
    if any(t in low for t in triggers) and "?" in text:
        return text.strip()
    return None


def build_augmented_prompt(user: str, search_text: str, system: str) -> str:
    return (f"{system}\n\n<user> {user} </user>\n{search_text}\n"
            f"<guizmo> (digere le resultat ci-dessus avec ton avis, ton francais naturel) ")


def extract_search_query(generated: str) -> str | None:
    m = SEARCH_RE.search(generated)
    return m.group(1).strip() if m else None
