"""Dataset APIs with lazy torch imports so CPU-only data validation remains usable."""

_CORE_EXPORTS = {
    "GraphQADataset",
    "EncodedItem",
    "build_label_maps",
    "build_vocab",
    "collate",
    "graph_file",
    "inverse_maps",
    "make_graph_datasets",
    "move_batch",
    "read_graphs",
    "read_visual_supervision",
    "refs_file",
    "visual_supervision_file",
}

__all__ = sorted(_CORE_EXPORTS)


def __getattr__(name: str):
    if name in _CORE_EXPORTS:
        from . import graph_qa

        return getattr(graph_qa, name)
    raise AttributeError(name)
