from pulp import lpSum

from .base_objective import BaseObjective


class ObjectiveTotalVisit(BaseObjective):
    objective_name = "objective_total_visit"

    def _set_objective(self) -> None:
        """目的変数を設定する"""
        self.model += (
            lpSum(self.variable.y[i] for i in range(1, self.variable.len_locations + 1)),
            "Total_Visit",
        )
