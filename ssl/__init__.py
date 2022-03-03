from .distill import DistillHard, DistillSoft
from .pl import PseudoLabel
from .ssl import SSLObjective, NoSSL

__all__ = ["DistillHard", "DistillSoft", "PseudoLabel", "NoSSL"]
