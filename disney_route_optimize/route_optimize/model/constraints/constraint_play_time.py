from pulp import lpSum

from .base_constraints import BaseConstraint


class ConstraintPlayTime(BaseConstraint):
    constraint_name = "constraint_play_time"

    def _set_constraint(self) -> None:
        """制約を設定する"""
        for i in range(1, self.variable.len_locations + 1):
            self.model += (
                lpSum(self.variable.u[i, j] for j in range(1, self.variable.len_locations + 1)) <= self.minutes * 60,
                f"Play_Time_Const({i})",
            )
