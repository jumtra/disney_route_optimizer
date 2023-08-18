from abc import ABC
from typing import ClassVar

from pulp import LpProblem

from ..variable import Variable


class BaseObjective(ABC):
    objective_name: ClassVar[str]

    def __init__(self, model: LpProblem, variable: Variable, list_rank: list[int], *args, **keargs) -> None:
        self.model: LpProblem = model
        self.variable: Variable = variable
        self.list_rank: list[int] = list_rank
        self._set_objective()

    def _set_objective(self) -> None:
        """目的変数を設定する"""
        raise NotImplementedError

    def get_model(self) -> LpProblem:
        return self.model
