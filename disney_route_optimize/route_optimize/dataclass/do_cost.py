from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from disney_route_optimize.common.config_manager import ConfigManager
from disney_route_optimize.route_optimize.dataclass.do_time import AttractionTime

ATTRACTION_NAME = str

FILL_VALUE = 24 * 60 * 60


@dataclass
class CostMatrix:
    """待ち時間予測結果を保存するデータクラス
    df_pred = 行：待ち時間、列：アトラクション名に変換
    アトラクション名は正規化したものを使用
    """

    config_manager: ConfigManager
    df_pred: pd.DataFrame
    df_step: pd.DataFrame
    dict_attraction2time: dict[str, float]
    dict_attraction2rank: dict[str, int]
    list_target_cols: list[str]

    key_attraction: str = "attraction_name"
    key_date: str = "date"
    key_numtime: str = "numtime"
    key_from: str = "from_attraction"
    key_to: str = "to_attraction"
    key_step: str = "steps"

    def __post_init__(self):
        self._set_params()
        self.list_rank = self._get_list_rank()
        self.list_cost, self.list_wait, self.list_move = self.make_cost_matrix()

    def _set_params(self) -> None:
        self.visit_time = self.config_manager.config.common.visit_time
        self.return_time = self.config_manager.config.common.return_time
        self.step_time = self.config_manager.config.cost.step_time
        self.sep_time = self.config_manager.config.cost.sep_time
        self.buffer_time = self.config_manager.config.cost.buffer_time

    def _get_list_rank(self) -> list[int]:
        self.list_target_cols = sorted(self.list_target_cols)
        list_rank = [self.dict_attraction2rank[attraction] for attraction in self.list_target_cols]
        return list_rank

    def make_cost_matrix(self) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        list_cost = []
        list_wait = []
        list_move = []
        visit_num = self._get_timenum(self.visit_time)
        return_num = self._get_timenum(self.return_time)
        df_step = self.df_step.loc[self.df_step[self.key_from].isin(self.list_target_cols) & self.df_step[self.key_to].isin(self.list_target_cols)]

        dict_col2idx = {col: idx for idx, col in enumerate(df_step.columns)}

        self.df_pred = self.df_pred.loc[:, self.df_pred.columns.isin(self.list_target_cols)]
        for i in range(visit_num, return_num):
            dict_pred = self.df_pred.query("numtime == @i")[list(self.df_pred.columns)].to_dict(orient="records")[0]

            dict_time = {}
            dict_wait = {}
            dict_move = {}

            for val in df_step.values:
                from_name = val[dict_col2idx[self.key_from]]
                to_name = val[dict_col2idx[self.key_to]]
                # アトラクションjの待ち時間 + iからjへの移動時間 + アトラクションjの所要時間 + バッファ
                time = (
                    val[dict_col2idx[self.key_step]] * self.step_time
                    + dict_pred.get(to_name, 0) * 60
                    + self.dict_attraction2time.get(to_name, 0)
                    + self.buffer_time
                )

                dict_time[(from_name, to_name)] = time
                dict_wait[(from_name, to_name)] = dict_pred.get(to_name, 0) * 60
                dict_move[(from_name, to_name)] = val[dict_col2idx[self.key_step]] * self.step_time

            keys = list(dict_time.keys())
            values = list(dict_time.values())
            # データフレームを作成
            df = pd.DataFrame({self.key_from: [key[0] for key in keys], self.key_to: [key[1] for key in keys], "cost": values})
            val = df.pivot(index=self.key_from, columns=self.key_to, values="cost").sort_index().T.sort_index().T.fillna(FILL_VALUE).values
            list_cost.append(val)

            # list_wait
            keys = list(dict_wait.keys())
            values = list(dict_wait.values())
            # データフレームを作成
            df = pd.DataFrame({self.key_from: [key[0] for key in keys], self.key_to: [key[1] for key in keys], "cost": values})
            val = df.pivot(index=self.key_from, columns=self.key_to, values="cost").sort_index().T.sort_index().T.fillna(FILL_VALUE).values
            list_wait.append(val)

            # list_move
            keys = list(dict_move.keys())
            values = list(dict_move.values())
            # データフレームを作成
            df = pd.DataFrame({self.key_from: [key[0] for key in keys], self.key_to: [key[1] for key in keys], "cost": values})
            val = df.pivot(index=self.key_from, columns=self.key_to, values="cost").sort_index().T.sort_index().T.fillna(FILL_VALUE).values
            list_move.append(val)
        return list_cost, list_wait, list_move

    def _get_timenum(self, time: str) -> int:
        base_time = pd.to_datetime("9:00")
        target_time = pd.to_datetime(time)
        time_num = int((target_time - base_time) / timedelta(days=1) * 24 * 60 / self.sep_time)
        return time_num
