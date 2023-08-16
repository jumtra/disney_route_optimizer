from pulp import lpSum

from .base_constraints import BaseConstraint


class ConstraintGlobalOnce(BaseConstraint):
    constraint_name = "constraint_global_once"

    def _set_constraint(self) -> None:
        """制約を設定する"""

        for t in range(self.variable.len_times):
            self.model += (
                lpSum(
                    [
                        self.variable.x[i, j, t]
                        for i in range(1, self.variable.len_locations + 1)
                        for j in range(1, self.variable.len_locations + 1)
                        if i != j
                    ]
                )
                <= 1,
                f"Time({t})",
            )
