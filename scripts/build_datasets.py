"""Build 48h : sang local (500 Vie + 200 Raison) + vrais FR ouverts SANS login.
Modes :
  python -m scripts.build_datasets
    -> 100% local (rapide, ton PC). pretrain.txt 1.6 Mo.
  python -m scripts.build_datasets --with-hf
    -> + Wikipedia FR (propre) + OSCAR-2301 FR + OASST FR. Si CulturaX
       gated echec -> Wikipedia/OSCAR prennent le relais (ton or 300 Mo).
  python -m scripts.build_datasets --with-hf --hf-gb 0.05 --max-docs 2000
    -> petit test HF (2 min).
  Sur Colab/Kaggle (recommande) : --with-hf --hf-gb 1.0 --max-docs 300000.
"""
import json
from pathlib import Path
from .gen_seed import gen_vie, gen_raison, gen_chercheur


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    import argparse
    import shutil
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-hf", action="store_true")
    ap.add_argument("--culturax-gb", type=float, default=1.0)
    ap.add_argument("--hf-gb", type=float, default=None, help="alias : Go de corpus FR ouvert (wiki+oscar)")
    ap.add_argument("--max-docs", type=int, default=300000)
    ap.add_argument("--source", default="auto", choices=["auto", "wiki", "oscar", "culturax"],
                    help="auto = wiki puis oscar puis culturax (si login)")
    ap.add_argument("--keep-hf", action="store_true",
                    help="ne PAS ecraser data/raw/hf_fr.txt et data/seed/vie.jsonl si HF echoue (garde le precedent)")
    args = ap.parse_args()
    corpus_gb = args.hf_gb if args.hf_gb is not None else args.culturax_gb
    base = Path("data")
    vie = gen_vie(500)
    raison = gen_raison(200)
    chercheur = gen_chercheur(80)
    extra_oasst: list[dict] = []
    hf_corpus: list[str] = []
    hf_source = "aucun"
    if args.with_hf:
        from .hf_loaders import load_culturax_fr, load_oasst2_fr, load_wikipedia_fr, load_oscar_fr, save_lines
        print(">> HF : OASST FR...")
        try:
            extra_oasst = load_oasst2_fr()
        except Exception as e:
            print(f"OASST echec : {e} (on continue sans)")
        print(f">> HF : corpus FR ouvert ({corpus_gb} Go, source={args.source})...")
        if args.source in ("auto", "wiki"):
            try:
                hf_corpus = load_wikipedia_fr(limit_chars=corpus_gb * 1e9, max_docs=args.max_docs)
                hf_source = "wikipedia-fr"
                save_lines(base / "raw" / "hf_fr.txt", hf_corpus)
            except Exception as e:
                print(f"Wikipedia echec : {e}")
        if not hf_corpus and args.source in ("auto", "oscar"):
            try:
                hf_corpus = load_oscar_fr(limit_chars=corpus_gb * 1e9, max_docs=args.max_docs)
                hf_source = "oscar-fr"
                save_lines(base / "raw" / "hf_fr.txt", hf_corpus)
            except Exception as e:
                print(f"OSCAR echec : {e}")
        if not hf_corpus and args.source in ("auto", "culturax"):
            try:
                hf_corpus = load_culturax_fr(limit_gb=corpus_gb, max_docs=args.max_docs)
                hf_source = "culturax-fr (gated, login requis)"
                save_lines(base / "raw" / "hf_fr.txt", hf_corpus)
            except Exception as e:
                print(f"CulturaX echec (gated, fais huggingface-cli login) : {e}")
        if not hf_corpus:
            print("Aucun corpus HF -> on garde le sang local seul (pense a lancer sur Colab/Kaggle).")
    vie_full = vie + extra_oasst
    hf_path = base / "raw" / "hf_fr.txt"
    # --keep-hf : si HF a echoue mais un precedent corpus existe, on le recharge
    # au lieu d'ecraser le precedent or (evite de perdre les 249 OASST + wiki).
    if args.with_hf and getattr(args, "keep_hf", False) and not hf_corpus and hf_path.exists():
        hf_corpus = [l for l in hf_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        hf_source = "precedent-garde"
        print(f"keep-hf : {len(hf_corpus)} lignes rechargees depuis {hf_path} (rien ecrase)")
    if not (args.with_hf and getattr(args, "keep_hf", False) and not extra_oasst):
        write_jsonl(base / "seed" / "vie.jsonl", vie_full)
    elif not (base / "seed" / "vie.jsonl").exists():
        write_jsonl(base / "seed" / "vie.jsonl", vie_full)
    else:
        print(f"keep-hf : vie.jsonl garde ({sum(1 for _ in (base / 'seed' / 'vie.jsonl').open(encoding='utf-8'))} lignes)")
        # restaure extra_oasst depuis le fichier garde pour le log
        try:
            import json as _json
            _old = [_json.loads(l) for l in (base / "seed" / "vie.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
            vie_full = _old
        except Exception:
            pass
    write_jsonl(base / "seed" / "raison.jsonl", raison)
    write_jsonl(base / "seed" / "chercheur.jsonl", chercheur)
    p = base / "processed" / "pretrain.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for r in raison:
        lines.append(" ".join((r["user"] + " " + r["guizmo"]).split()))
    for r in vie[:100]:
        lines.append(" ".join((r["user"] + " " + r["guizmo"]).split()))
    lines += hf_corpus
    seed_repeat = lines[:300] * 4 if hf_corpus else lines * 20
    all_lines = seed_repeat + hf_corpus if hf_corpus else seed_repeat
    with p.open("w", encoding="utf-8") as f:
        for t in all_lines:
            f.write(t + "\n")
    size_mo = p.stat().st_size / 1e6
    (base / "raw").mkdir(parents=True, exist_ok=True)
    (base / "raw" / "cache_info.txt").write_text(
        f"vie={len(vie_full)} (gen=500 oasst={len(extra_oasst)}) raison={len(raison)} "
        f"chercheur={len(chercheur)} hf={len(hf_corpus)} source={hf_source} Mo={size_mo:.1f}\n", encoding="utf-8")
    print(f"OK: vie={len(vie_full)} raison={len(raison)} chercheur={len(chercheur)} hf={len(hf_corpus)} [{hf_source}] -> {p} ({size_mo:.1f} Mo)")


if __name__ == "__main__":
    main()
