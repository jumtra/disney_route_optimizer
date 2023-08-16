from pulp import lpSum

from .base_constraints import BaseConstraint


class ConstraintGlobalTime(BaseConstraint):
    constraint_name = "constraint_global_time"

    def _set_constraint(self) -> None:
        """制約を設定する"""

        for j in range(1, self.variable.len_locations + 1):
            self.model += (
                lpSum(self.variable.u[j, i] for i in range(1, self.variable.len_locations + 1) if i != j)
                - lpSum(
                    self.variable.x[i, j, t] * self.sep_time * t * 60
                    for t in range(self.variable.len_times)
                    for i in range(1, self.variable.len_locations + 1)
                    if i != j
                )
                <= 0,
                f"GlobalTime_{j}",
            )
