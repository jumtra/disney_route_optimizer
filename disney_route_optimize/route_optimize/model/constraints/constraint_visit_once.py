from pulp import lpSum

from .base_constraints import BaseConstraint


class ConstraintVisitOnce(BaseConstraint):
    constraint_name = "constraint_visit_once"

    def _set_constraint(self) -> None:
        """制約を設定する"""

        for i in range(1, self.variable.len_locations + 1):
            self.model += (
                lpSum([self.variable.x[i, j, t] for j in range(1, self.variable.len_locations + 1) for t in range(self.variable.len_times) if i != j])
                == self.variable.y[i],
                f"In({i})",
            )
            self.model += (
                lpSum([self.variable.x[j, i, t] for j in range(1, self.variable.len_locations + 1) for t in range(self.variable.len_times) if i != j])
                == self.variable.y[i],
                f"Out({i})",
            )
