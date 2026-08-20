"""TennisVAR: stroke-evidence-grounded tactical reasoning for tennis video."""

__version__ = "0.1.0"
__all__ = ["TennisVAR"]


def __getattr__(name: str):
    if name == "TennisVAR":
        from .pipeline import TennisVAR

        return TennisVAR
    raise AttributeError(name)
