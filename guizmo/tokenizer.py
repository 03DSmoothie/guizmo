"""Tokenizer FR de Guizmo : BPE byte-level (style GPT-2 / Llama) + tokens speciaux.
Tokens metier : <user> <guizmo> <search> </search> <result> </result>
"""
import json
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

SPECIAL = ["<pad>", "<s>", "</s>", "<unk>", "<mask>", "<user>", "<guizmo>", "<search>", "</search>", "<result>", "</result>"]
IDS = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "<mask>": 4}


def train_tokenizer(corpus_files: list[str], out_dir: str, vocab_size: int = 32000):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = ByteLevelBPETokenizer(add_prefix_space=True)
    tok.train(files=corpus_files, vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL)
    tok.save_model(str(out))
    # Sauvegarde aussi le tokenizer complet (chargement robuste, anti mismatch vocab/merges)
    # Note: ByteLevelBPETokenizer n'a pas de .save() qui marche bien -> on utilise le backend.
    try:
        tok._tokenizer.save(str(out / "tokenizer.json"))
    except Exception as e:
        print(f"warn: pas de tokenizer.json ({e})")
    real = tok.get_vocab_size()
    cfg = {"vocab_size": vocab_size, "vocab_reel": real, "special": SPECIAL, "ids": IDS, "type": "ByteLevelBPE"}
    (out / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"vocab demande={vocab_size} reel={real} (reel plafonne par la taille du corpus, normal en demo)")
    return str(out / "vocab.json"), str(out / "merges.txt")


class GuizmoTokenizer:
    """Wrapper minimal autour du tokenizer HF `tokenizers` pour train/infer."""

    def __init__(self, vocab_path: str, merges_path: str):
        from tokenizers import ByteLevelBPETokenizer, Tokenizer
        from pathlib import Path as _P
        # 1) Si tokenizer.json existe a cote (paire coherente), on le prefere :
        #    ca evite le mismatch vocab.json/merges.txt (ex: `čĊ` out of vocabulary).
        tj = str(_P(vocab_path).parent / "tokenizer.json")
        bt = None
        if _P(tj).exists():
            try:
                raw = Tokenizer.from_file(tj)
                bt = ByteLevelBPETokenizer(add_prefix_space=True)
                bt._tokenizer = raw
            except Exception:
                bt = None
        if bt is None:
            bt = ByteLevelBPETokenizer(vocab_path, merges_path)
        bt.add_special_tokens(SPECIAL)  # sinon <user> est coupe en < user >
        self._tok = bt
        self._bt = bt
        self.vocab = bt.get_vocab()
        self.pad_id = self.vocab.get("<pad>", 0)
        self.bos_id = self.vocab.get("<s>", 1)
        self.eos_id = self.vocab.get("</s>", 2)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> list[int]:
        ids = self._tok.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)

    def encode_dialog(self, user: str, guizmo: str) -> list[int]:
        return self.encode(f"<user> {user} <guizmo> {guizmo}", add_bos=True, add_eos=True)
