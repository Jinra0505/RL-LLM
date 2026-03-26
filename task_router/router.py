from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTaskRouter(ABC):
    @abstractmethod
    def route(self, summary: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
