"""RallyMotionReasoner: stroke-evidence-grounded tactical reasoning for tennis video."""

__version__ = "0.1.0"
__all__ = ["RallyMotionReasoner"]


def __getattr__(name: str):
    if name == "RallyMotionReasoner":
        from .pipeline import RallyMotionReasoner

        return RallyMotionReasoner
    raise AttributeError(name)
