from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(root: Path, *, pattern: str = "*.pt") -> str:
    """Hash filenames and bytes for an immutable cache directory."""
    root = Path(root)
    paths = sorted(path for path in root.rglob(pattern) if path.is_file()) if root.is_dir() else []
    if not paths:
        raise FileNotFoundError(f"no files matching {pattern!r} under {root}")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()
