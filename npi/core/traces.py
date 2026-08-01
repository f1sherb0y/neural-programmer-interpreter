from dataclasses import dataclass, field
from typing import Any

from npi.core.spec import Arguments


@dataclass(frozen=True)
class Decision:
    observation: Any
    end: bool
    next_program: int
    next_arguments: Arguments
    has_child: bool = True


@dataclass
class Episode:
    program: int
    arguments: Arguments
    decisions: list[Decision] = field(default_factory=list)
