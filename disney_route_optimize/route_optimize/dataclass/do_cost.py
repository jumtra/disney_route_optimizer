from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from disney_route_optimize.common.config_maneger import ConfigManeger
from disney_route_optimize.route_optimize.dataclass.do_time import AttractionTime

ATTRACTION_NAME = str


@dataclass
class CostMatrix:
    """待ち時間予測結果を保存するデータクラス
    df_pred = 行：待ち時間、列：アトラクション名に変換
    アトラクション名は正規化したものを使用
    """

    config_maneger: ConfigManeger
    df_pred: pd.DataFrame
    df_step: pd.DataFrame
    dict_attraction2time: dict[str, float]
    list_target_cols: list[str]

    key_attraction: str = "attraction_name"
    key_date: str = "date"
    key_numtime: str = "numtime"
    key_from: str = "from_attraction"
    key_to: str = "to_attraction"
    key_step: str = "steps"

    def __post_init__(self):
        self._set_params()
        self.list_cost = self.make_cost_matrix()

    def _set_params(self) -> None:
        self.visit_time = self.config_maneger.config.common.visit_time
        self.return_time = self.config_maneger.config.common.return_time
        self.step_time = self.config_maneger.config.cost.step_time
        self.sep_time = self.config_maneger.config.cost.sep_time
        self.buffer_time = self.config_maneger.config.cost.buffer_time

    def make_cost_matrix(self) -> list[np.ndarray]:
        list_cost = []
        visit_num = self._get_timenum(self.visit_time)
        return_num = self._get_timenum(self.return_time)
        df_step = self.df_step.loc[self.df_step[self.key_from].isin(self.list_target_cols) & self.df_step[self.key_to].isin(self.list_target_cols)]

        # df_step["movetime"] = df_step[self.key_step] * self.step_time  # 1歩で0.5秒 1分間120歩 歩幅63センチらしい
        dict_col2idx = {col: idx for idx, col in enumerate(df_step.columns)}

        self.df_pred = self.df_pred.loc[:, self.df_pred.columns.isin(self.list_target_cols)]
        for i in range(visit_num, return_num + 1):
            dict_pred = self.df_pred.query("numtime == @i")[list(self.df_pred.columns)].to_dict(orient="records")[0]

            dict_time = {}

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
                time = (
                    val[dict_col2idx[self.key_step]] * self.step_time
                    + dict_pred.get(from_name, 0) * 60
                    + self.dict_attraction2time.get(from_name, 0)
                    + self.buffer_time
                )
                dict_time[(to_name, from_name)] = time
            keys = list(dict_time.keys())
            values = list(dict_time.values())

            # データフレームを作成
            df = pd.DataFrame({self.key_from: [key[0] for key in keys], self.key_to: [key[1] for key in keys], "cost": values})
            val = df.pivot(index=self.key_from, columns=self.key_to, values="cost").sort_index().T.sort_index().fillna(10000).values

            list_cost.append(val)
        return list_cost

    def _get_timenum(self, time: str) -> int:
        base_time = pd.to_datetime("9:00")
        target_time = pd.to_datetime(time)
        time_num = int((target_time - base_time) / timedelta(days=1) * 24 * 60 / self.sep_time)
        return time_num
