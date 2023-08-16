from abc import ABC
from typing import ClassVar

from pulp import LpProblem

from ..variable import Variable


class BaseConstraint(ABC):
    constraint_name: ClassVar[str]

    def __init__(self, model: LpProblem, variable: Variable, sep_time: int, global_time: float, *args, **keargs) -> None:
        self.sep_time: int = sep_time
        self.minutes: float = global_time
        self.model: LpProblem = model
        self.variable: Variable = variable
        self._set_constraint()

    def _set_constraint(self) -> None:
        """制約を設定する"""
        raise NotImplementedError

    def get_model(self) -> LpProblem:
        return self.model
