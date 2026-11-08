"""Small process-local caches for memory-limited serving instances."""
from collections import OrderedDict
from collections.abc import Callable, Hashable
from threading import Lock
from typing import Generic, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class BoundedCache(Generic[K, V]):
    def __init__(self, maxsize: int):
        if maxsize < 1:
            raise ValueError("cache size must be positive")
        self.maxsize = maxsize
        self._items: OrderedDict[K, V] = OrderedDict()
        self._lock = Lock()

    def get_or_create(self, key: K, factory: Callable[[], V]) -> V:
        # Loading under the lock also prevents parallel first requests from
        # building multiple copies of a large session or an expensive lap fit.
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
            value = factory()
            self._items[key] = value
            if len(self._items) > self.maxsize:
                self._items.popitem(last=False)
            return value

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
