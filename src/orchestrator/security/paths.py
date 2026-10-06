from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union


def normalize_path(path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve and normalize a filesystem path against base_dir (default: current working directory).
    Handles relative and absolute paths, trailing separators, '.', and '..'.
    Raises ValueError on invalid path strings (e.g. containing null bytes or empty).
    """
    if isinstance(path, str):
        if not path or not path.strip():
            raise ValueError("Path must not be empty or blank")
        if "\0" in path:
            raise ValueError("Path must not contain null bytes")
        raw_path = Path(path.strip())
    elif isinstance(path, Path):
        raw_path = path
    else:
        raise ValueError(f"Expected str or Path, got {type(path).__name__}")

    base = Path(base_dir).resolve() if base_dir is not None else Path.cwd().resolve()
    if raw_path.is_absolute():
        return raw_path.resolve()
    return (base / raw_path).resolve()


def is_subpath(
    target_path: Union[str, Path],
    allowed_scope: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
) -> bool:
    """
    Deterministically check whether target_path resides within or equals allowed_scope.
    Strictly prevents prefix ambiguity (e.g. /project/foo vs /project/foobar),
    normalizes trailing slashes, and resolves traversal ('..') safely.
    """
    try:
        norm_target = normalize_path(target_path, base_dir=base_dir)
        norm_scope = normalize_path(allowed_scope, base_dir=base_dir)
    except Exception:
        return False

    # 1. Native pathlib relative containment check
    try:
        if norm_target.is_relative_to(norm_scope):
            return True
    except (ValueError, AttributeError):
        pass

    # 2. Case-insensitive / normalized true boundary check
    target_str = str(norm_target).lower().replace("\\", "/").rstrip("/")
    scope_str = str(norm_scope).lower().replace("\\", "/").rstrip("/")

    if target_str == scope_str:
        return True

    return target_str.startswith(scope_str + "/")


def is_path_allowed(
    target_path: str,
    allowed_paths: List[str],
    base_dir: Optional[Union[str, Path]] = None,
) -> bool:
    """
    Check if target_path is permitted according to allowed_paths scope.
    If allowed_paths is empty, access is unrestricted (returns True).
    If allowed_paths is non-empty, target_path must be contained within at least one allowed scope.
    """
    if not allowed_paths:
        return True

    return any(is_subpath(target_path, p, base_dir=base_dir) for p in allowed_paths)


def validate_path_string(path: str) -> None:
    """Validate that a single path string conforms to safe path syntax."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Path must be a non-empty string")
    if "\0" in path:
        raise ValueError("Path contains invalid null byte")
