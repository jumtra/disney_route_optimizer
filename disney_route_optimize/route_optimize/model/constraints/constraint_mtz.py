from pulp import lpSum

from .base_constraints import BaseConstraint


class ConstraintMTZ(BaseConstraint):
    constraint_name = "constraint_mtz"

    def _set_constraint(self) -> None:
        """制約を設定する"""

        for j in range(2, self.variable.len_locations + 1):
            self.model += (
                lpSum(
                    self.variable.u[i, j]
                    + lpSum(self.variable.cost_list[t][i - 1][j - 1] * self.variable.x[i, j, t] for t in range(self.variable.len_times))
                    for i in range(1, self.variable.len_locations + 1)
                    if i != j
                )
                - lpSum(self.variable.u[j, k] for k in range(1, self.variable.len_locations + 1) if k != j)
                <= 0,
                f"MTZ_{j}",
            )
