"""Synthese V2 : transforme resultats web -> reponse Guizmo.
Regle d'or : JAMAIS de recrachage. Toujours : INTENTION comprise + CONTEXTE
+ 2-3 faits digeres + avis tranche + relance. Si pas de resultats :
dis-le + aide quand meme (jamais d'invention).
"""
from .router import Route


SYS_TON = ("Tu es Guizmo, pote francais : empathie d'abord, avis tranche, "
           "jamais robot, toujours une relance.")


def synth_prompt(user, rt, search_block):
    lines = [SYS_TON, "", f"Question : {user}",
             f"Intention : {rt.intent} | Contexte : {rt.context_summary or '-'}",
             f"Requetes web : {' ; '.join(rt.queries) if rt.queries else '-'}"]
    if search_block:
        lines += ["", "Ce que le web dit :", search_block, "",
                  "Consigne : 1) montre que tu as compris l'intention et le contexte, "
                  "2) digere 2-3 faits en 3-5 phrases avec TON avis, jamais de copier-coller, "
                  "3) termine par une relance."]
    else:
        lines += ["", "Pas de recherche : reponds direct, court, avec avis + relance."]
    return "\n".join(lines) + "\n<guizmo> "


def fallback_answer(user, rt, n_snippets=0):
    low = user.lower()
    if rt.intent == "salutation":
        return "Wesh, ça va ? Raconte-moi, c'est quoi ton sujet du moment ?"
    if rt.intent == "emotion" and not rt.needs_search:
        return ("Je comprends, ça pèse. Raconte-moi ce qui s'est passé exactement, "
                "on démêle ça ensemble. Tu veux un plan concret ou juste vider ton sac ?")
    if n_snippets == 0 and rt.needs_search:
        return ("Hmm, ma recherche a rien donné de bon là. Reformule avec un mot-clé "
                "en plus (ville, année, modèle) et je re-check direct. "
                "C'est quoi le détail qui compte le plus pour toi ?")
    return ("Je check ça et je te résume avec mon avis dans 2 secondes. "
            "En attendant : c'est pour décider quoi exactement ?")
