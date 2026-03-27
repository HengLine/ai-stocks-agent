from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class AgentResult:
    name: str
    success: bool
    data: Dict[str, Any]
    error: str | None = None


class BaseAgent:
    name: str = "base"

    def run(self, params: Dict[str, Any], context: Dict[str, Any]) -> AgentResult:
        raise NotImplementedError


