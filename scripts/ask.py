"""Une question -> une reponse Guizmo V3 (pour tester vite, ici ou ailleurs).
Usage: python -m scripts.ask "C'est quoi la tetanique ?"
Sans argument : demande la question au clavier.
"""
import sys
from guizmo.pipeline import answer


def main():
    q = " ".join(sys.argv[1:]).strip()
    if not q:
        q = input("Toi : ").strip()
    if not q:
        return
    txt, rt = answer(q)
    print(f"[{rt.intent} | search={rt.needs_search} | q={'; '.join(rt.queries) or '-'}]")
    print(txt)


if __name__ == "__main__":
    main()