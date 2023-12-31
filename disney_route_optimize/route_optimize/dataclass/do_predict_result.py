from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from disney_route_optimize.common.config_manager import ConfigManager

from ..preprocess.clean_data import make_clean_master

ATTRACTION_NAME = str


@dataclass
class PredictResult:
    """待ち時間予測結果を保存するデータクラス
    df_pred = 行：待ち時間、列：アトラクション名に変換
    アトラクション名は正規化したものを使用
    """

    config_manager: ConfigManager
    df_pred: pd.DataFrame = field(default_factory=pd.DataFrame)
    key_attraction: str = "attraction_name"
    key_date: str = "date"
    key_numtime: str = "numtime"
    key_renamed_numtime: str = "numtime"

    def __post_init__(self):
        df_pred = pd.read_csv(
            Path(self.config_manager.config.output.wp_output.path_predict_dir) / self.config_manager.config.output.wp_output.pred_test_file
        )
        visit_date = self.config_manager.config.common.visit_date
        df_pred = df_pred.query("date == @visit_date")
        df = self.clean_data(df=df_pred)
        df = df.sort_values(by=[self.key_attraction, self.key_date, self.key_numtime])

        # 縦持ちに変換
        df = df.pivot(index=[self.key_date, self.key_numtime], columns=self.key_attraction, values="pred").reset_index()
        self.df_pred = df.set_index([self.key_date, self.key_numtime])

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """アトラクション名を正規化"""
        df = make_clean_master(df=df, key_attraction=self.key_attraction)
        return df
