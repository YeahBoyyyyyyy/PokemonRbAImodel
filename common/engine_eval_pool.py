"""Pool of :class:`EngineTurnEvaluator` instances for parallel candidate eval.

Each evaluator owns a dedicated :class:`PkmnBridge` subprocess so ``step``
calls are not serialized on a single Node process lock.
"""

from __future__ import annotations

import queue
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Callable, Iterable, List, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class EngineEvalPool:
    """Checkout / return evaluators across worker threads."""

    def __init__(self, evaluators: Iterable[object]) -> None:
        self._evaluators = list(evaluators)
        self._free: queue.Queue[object] = queue.Queue()
        for ev in self._evaluators:
            self._free.put(ev)

    @property
    def size(self) -> int:
        return len(self._evaluators)

    @contextmanager
    def borrow(self):
        ev = self._free.get()
        try:
            yield ev
        finally:
            self._free.put(ev)

    def map(self, fn: Callable[[object, T], R], items: Iterable[T]) -> List[R]:
        """Run ``fn(evaluator, item)`` for each item, using up to ``size`` threads."""
        items_list = list(items)
        if not items_list:
            return []
        if self.size <= 1:
            ev = self._evaluators[0]
            return [fn(ev, item) for item in items_list]

        def run_one(item: T) -> R:
            with self.borrow() as ev:
                return fn(ev, item)

        max_workers = min(self.size, len(items_list))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(run_one, items_list))
