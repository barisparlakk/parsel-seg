"""Command-line overrides for nested YAML configs."""

from typing import Any

import yaml


def _parse(raw: str) -> Any:
    """int, then float (YAML 1.1 reads "3e-4" as a string), then any other YAML value."""
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return yaml.safe_load(raw)


def apply_overrides(cfg: dict, items: list[str]) -> dict:
    """Apply "a.b.c=value" overrides in place.

    Values are parsed as int, float or YAML (null, true, strings).

    Every intermediate key must already exist, so a typo in a section name fails loudly. The
    last key may be new.
    """
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override {item!r} must look like section.key=value")
        path, raw = item.split("=", 1)
        keys = path.split(".")
        node: Any = cfg
        for key in keys[:-1]:
            if not isinstance(node, dict) or key not in node:
                raise KeyError(f"Override {item!r}: no section {key!r} in the config")
            node = node[key]
        if not isinstance(node, dict):
            raise KeyError(f"Override {item!r}: {'.'.join(keys[:-1])!r} is not a section")
        node[keys[-1]] = _parse(raw)
    return cfg
