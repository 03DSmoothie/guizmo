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


def test_router_v2():
    from guizmo.router import route
    assert route("Salut").intent == "salutation"
    assert route("Salut").needs_search is False
    assert route("Je suis triste ce soir").intent == "emotion"
    assert route("C'est quoi le prix du bitcoin aujourd'hui ?").intent == "actu"
    r = route("iPhone ou Samsung ?", None)
    assert r.intent == "comparaison" and len(r.queries) >= 2
    r2 = route("Et lui ?", [{"user": "Qui a gagne la coupe du monde 2022 ?"}])
    assert r2.intent == "suivi" and "coupe du monde" in r2.queries[0].lower()
    # V2 : cherche systematiquement sauf social/emotion pure
    assert route("Qui a ecrit Les Miserables ?").needs_search is True


def test_pipeline_no_hallucination():
    from guizmo.pipeline import answer
    txt, rt = answer("Salut")
    assert "wesh" in txt.lower()
    assert rt.needs_search is False
