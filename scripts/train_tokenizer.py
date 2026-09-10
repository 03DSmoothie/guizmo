"""Entraine le tokenizer FR sur ton corpus : python -m scripts.train_tokenizer --corpus data/raw/*.txt"""
import argparse
import glob
from guizmo.tokenizer import train_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", required=True, help="fichiers .txt (glob accepte)")
    ap.add_argument("--out", default="tokenizer")
    ap.add_argument("--vocab_size", type=int, default=32000)
    args = ap.parse_args()
    files: list[str] = []
    for pat in args.corpus:
        files += glob.glob(pat)
    assert files, "aucun fichier corpus trouve"
    print(f"{len(files)} fichiers -> train BPE {args.vocab_size}")
    v, m = train_tokenizer(files, args.out, args.vocab_size)
    print(f"OK : {v} {m}")


if __name__ == "__main__":
    main()
