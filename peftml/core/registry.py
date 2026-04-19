"""Lightweight registry for extensible component lookup."""

from __future__ import annotations

from typing import Any, Callable, Dict, TypeVar

T = TypeVar("T")


class Registry:
    """A name→callable registry with collision detection.

    Usage::

        PRUNERS = Registry("pruner")

        @PRUNERS.register("my_pruner")
        class MyPruner: ...

        pruner_cls = PRUNERS["my_pruner"]
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: Dict[str, Any] = {}

    def register(self, key: str) -> Callable[[T], T]:
        """Decorator that registers *cls_or_fn* under *key*."""

        def decorator(cls_or_fn: T) -> T:
            if key in self._registry:
                raise KeyError(
                    f"'{key}' is already registered in the {self.name!r} registry "
                    f"(existing: {self._registry[key]})."
                )
            self._registry[key] = cls_or_fn
            return cls_or_fn

        return decorator

    def __getitem__(self, key: str) -> Any:
        if key not in self._registry:
            available = ", ".join(sorted(self._registry)) or "(empty)"
            raise KeyError(
                f"'{key}' not found in the {self.name!r} registry. "
                f"Available: {available}"
            )
        return self._registry[key]

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def keys(self):
        return self._registry.keys()

    def __repr__(self) -> str:
        entries = ", ".join(sorted(self._registry))
        return f"Registry({self.name!r}, [{entries}])"
