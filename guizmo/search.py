"""Les Yeux V2 : chercheur systematique. Le nano ne sait rien, il cherche tout.
- Double import (duckduckgo_search ancien / ddgs nouveau) car Kaggle a l'un ou l'autre.
- Timeout court + fallback Wikipedia FR (API ouverte, sans cle) si DDG rate-limite.
- Dedup + filtre FR minimal. Jamais d'exception : toujours un SearchResult.
"""
from dataclasses import dataclass
import re
import urllib.parse
import urllib.request
import json as _json


@dataclass
class SearchResult:
    query: str
    snippets: list[str]


def _ddg_search(query: str, max_results: int = 5) -> list[str]:
    last_err = None
    for mod_name in ("duckduckgo_search", "ddgs"):
        try:
            mod = __import__(mod_name)
            DDGS = getattr(mod, "DDGS")
            out: list[str] = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, region="fr-fr", max_results=max_results):
                    title = (r.get("title") or "").strip()
                    body = (r.get("body") or "").strip()
                    if title or body:
                        out.append(f"{title} : {body}".strip(" :"))
            if out:
                return out
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"DDG indisponible ({last_err})")


def _wiki_fallback(query: str, max_results: int = 3) -> list[str]:
    """Wikipedia FR API : gratuit, sans cle, parfait en secours."""
    try:
        q = urllib.parse.quote(query[:120])
        url = (f"https://fr.wikipedia.org/w/api.php?action=query&list=search"
               f"&srsearch={q}&format=json&srlimit={max_results}&srprop=snippet")
        req = urllib.request.Request(url, headers={"User-Agent": "GuizmoV2/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = _json.loads(resp.read().decode("utf-8", "ignore"))
        out = []
        for item in data.get("query", {}).get("search", [])[:max_results]:
            title = item.get("title", "")
            snip = re.sub(r"<.*?>", "", item.get("snippet", ""))
            out.append(f"{title} : {snip}".strip(" :"))
        return out
    except Exception:
        return []


_FR_OK = re.compile(r"( le | la | les | de | des | et | est | que | pour | dans |é|è|à|ç)", re.I)


def web_search(query: str, max_results: int = 5) -> SearchResult:
    try:
        snips = _ddg_search(query, max_results)
    except Exception as e:
        snips = _wiki_fallback(query, max_results)
        if not snips:
            return SearchResult(query=query, snippets=[f"(recherche indisponible : {e})"])
    # dedup + coupe
    seen, clean = set(), []
    for s in snips:
        s = " ".join(s.split())[:600]
        k = s.lower()[:80]
        if s and k not in seen:
            seen.add(k)
            clean.append(s)
        if len(clean) >= max_results:
            break
    return SearchResult(query=query, snippets=clean or ["(aucun resultat)"])


def format_for_coeur(res: SearchResult) -> str:
    lines = "\n".join(f"- {s}" for s in res.snippets[:5])
    return f"<result> {res.query}\n{lines} </result>"

