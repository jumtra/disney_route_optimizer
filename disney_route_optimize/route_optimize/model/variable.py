from dataclasses import dataclass, field

import numpy as np
from pulp import LpBinary, LpContinuous, LpVariable


@dataclass
class Variable:
    """最適化の変数群"""

    cost_list: list[np.ndarray]
    len_locations: int
    len_times: int
    # x: dict[tuple(int, int, int), LpVariable] = field(default=dict)
    # y: dict[int, LpVariable] = field(default=dict)
    # u: dict[tuple(int, int), LpVariable] = field(default=dict)

    def __post_init__(self):
        self.x = self._get_x()
        self.y = self._get_y()
        self.u = self._get_u()

    def _get_x(self):
        """アトラクションiからjに時刻tで巡回するかどうかを表す変数"""
        x = {}
        for i in range(1, self.len_locations + 1):
            for j in range(1, self.len_locations + 1):
                for t in range(self.len_times):
                    x[i, j, t] = LpVariable(f"x_{i}_{j}_{t}", cat=LpBinary)
        return x

    def _get_y(self):
        """各アトラクションiを巡回するかどうかの変数"""
        y = {}
        for i in range(1, self.len_locations + 1):
            y[i] = LpVariable(f"y_{i}", cat=LpBinary)
        return y

    def _get_u(self):
        """各アトラクションiからjに移動する時のアトラクションiの出発時刻を表す変数"""
        u = {}
        for i in range(1, self.len_locations + 1):
            for j in range(1, self.len_locations + 1):
                u[i, j] = LpVariable(f"u_{i}_{j}", cat=LpContinuous, lowBound=0)
        return u
