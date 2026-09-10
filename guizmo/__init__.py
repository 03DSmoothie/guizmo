"""Guizmo - le petit LLM francais qui ne sait rien, mais comprend tout."""
from .config import GuizmoConfig
from .model import GuizmoForCausalLM, RMSNorm

__all__ = ["GuizmoConfig", "GuizmoForCausalLM", "RMSNorm"]
__version__ = "0.1.0"
