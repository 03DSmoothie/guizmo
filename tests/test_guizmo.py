import torch
from guizmo.config import GuizmoConfig
from guizmo.model import GuizmoForCausalLM, RMSNorm
from guizmo.data import mix_datasets
from guizmo.converse import needs_search, extract_search_query


def test_config_params():
    cfg = GuizmoConfig.tiny()
    assert cfg.count_params() > 1_000_000


def test_forward_tiny():
    cfg = GuizmoConfig.tiny()
    cfg.vocab_size = 500
    m = GuizmoForCausalLM(cfg)
    x = torch.randint(0, 500, (2, 16))
    logits, loss = m(x, x)
    assert logits.shape == (2, 16, 500)
    assert loss.item() > 0


def test_rmsnorm():
    n = RMSNorm(8)
    x = torch.randn(2, 4, 8)
    assert n(x).shape == x.shape


def test_mix_ratio():
    vie = [{"user": "a", "guizmo": "b"}] * 60
    rai = [{"user": "a", "guizmo": "b"}] * 30
    che = [{"user": "a", "guizmo": "b"}] * 10
    m = mix_datasets(vie, rai, che, seed=0)
    assert len(m) == 100


def test_shifted_loss_sane():
    """Garde-fou : sans shift la loss triche (~0). Avec shift, init ~ ln(vocab)."""
    import math
    cfg = GuizmoConfig.tiny()
    cfg.vocab_size = 500
    m = GuizmoForCausalLM(cfg)
    torch.manual_seed(0)
    x = torch.randint(0, 500, (2, 16))
    _, loss = m(x, x)
    assert 4.0 < loss.item() < 8.0, f"loss suspecte {loss.item():.3f} (init attendue ~ln(500)={math.log(500):.2f})"


def test_search_detect():
    assert needs_search("C'est qui le dernier ballon d'or ?") is not None
    assert extract_search_query("bla <search>meteo Paris</search> bla") == "meteo Paris"
    assert needs_search("J'aime pas les maths") is None
