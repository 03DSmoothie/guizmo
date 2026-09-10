"""Les Yeux de Guizmo : recherche web simple (DuckDuckGo, gratuit, sans cle).
Si pas d'internet / pas de lib : repli gracieux -> pas de fait, le Coeur repond seul.
"""
from dataclasses import dataclass


@dataclass
class SearchResult:
    query: str
    snippets: list[str]


def web_search(query: str, max_results: int = 5) -> SearchResult:
    try:
        from duckduckgo_search import DDGS
        out: list[str] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, region="fr-fr", max_results=max_results):
                title = r.get("title", "")
                body = r.get("body", "")
                out.append(f"{title} : {body}".strip())
        return SearchResult(query=query, snippets=out or ["(aucun resultat)"])
    except Exception as e:  # offline / lib absente
        return SearchResult(query=query, snippets=[f"(recherche indisponible : {e})"])


def format_for_coeur(res: SearchResult) -> str:
    lines = "\n".join(f"- {s}" for s in res.snippets[:5])
    return f"<result> {res.query}\n{lines} </result>"
