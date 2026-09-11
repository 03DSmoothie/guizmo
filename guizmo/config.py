"""Config centralisee de Guizmo (Coeur 150M). Inspire Llama3 / Qwen3 / SmolLM2."""
from dataclasses import dataclass, field


@dataclass
class GuizmoConfig:
    vocab_size: int = 32000
    max_seq_len: int = 2048
    n_layer: int = 12
    n_embd: int = 768
    n_head: int = 12
    n_kv_head: int = 4
    intermediate_size: int = 2048
    rope_theta: float = 100000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd doit etre divisible par n_head"
        assert self.n_head % self.n_kv_head == 0, "GQA: n_head % n_kv_head == 0"
        self.head_dim = self.n_embd // self.n_head

    def count_params(self) -> int:
        """Estimation parametres (embeddings + 12 blocs)."""
        h, v, L, inter = self.n_embd, self.vocab_size, self.n_layer, self.intermediate_size
        tok = v * h
        per_block = 4 * h * h  # approx q,k,v,o avec GQA un peu moins
        mlp = 3 * h * inter    # SwiGLU: gate, up, down
        norms = 2 * h
        total = tok + L * (per_block + mlp + norms)
        if not self.tie_word_embeddings:
            total += v * h
        return total

    @classmethod
    def tiny(cls) -> "GuizmoConfig":
        """Version 30M pour tester sur CPU / Colab gratuit."""
        return cls(n_layer=6, n_embd=512, n_head=8, n_kv_head=2, intermediate_size=1024)

    @classmethod
    def nano(cls) -> "GuizmoConfig":
        """V2 chercheur systematique : ~8-12M, juste intention/contexte/reformulation.
        Pas de connaissances stockees : il comprend, cherche, synthetise.
        4 couches, 384 hidden, GQA 6Q/2KV, SwiGLU 768. Tourne sur CPU/Kaggle sans crash.
        """
        return cls(n_layer=4, n_embd=384, n_head=6, n_kv_head=2, intermediate_size=768)

    @classmethod
    def coeur150m(cls) -> "GuizmoConfig":
        return cls()
