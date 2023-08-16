from pulp import lpSum

from .base_constraints import BaseConstraint


class ConstraintTimeWindow(BaseConstraint):
    constraint_name = "constraint_time_window"

    def _set_constraint(self) -> None:
        """制約を設定する"""

        for i in range(1, self.variable.len_locations + 1):
            for j in range(1, self.variable.len_locations + 1):
                if i != j:
                    self.model += 0 <= self.variable.u[i, j], f"MinWindow({i}_{j})"
                    self.model += (
                        self.variable.u[i, j] <= self.minutes * 60 * lpSum(self.variable.x[i, j, t] for t in range(self.variable.len_times)),
                        f"MaxWindow({i}_{j})",
                    )
