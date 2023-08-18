from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from disney_route_optimize.common.config_maneger import ConfigManeger

from ..preprocess.clean_data import make_clean_master

ATTRACTION_NAME = str
TIME = float


@dataclass
class AttractionTime:
    """アトラクション別の稼働時間を保存するデータクラス
    アトラクション名は正規化したものを使用
    """

    config_maneger: ConfigManeger
    dict_attraction2time: dict[ATTRACTION_NAME, TIME] = field(default_factory=dict)
    target_attraction: list[ATTRACTION_NAME] = field(default_factory=list)
    key_attraction: str = "attraction"
    key_time: str = "time"

    def __post_init__(self):
        path_master = Path(self.config_maneger.config.input.path_master_file)
        df_master = pd.read_csv(path_master)
        df = self.clean_data(df_master)
        self.target_attraction = list(df[self.key_attraction].unique())
        dict_col2idx = {col: idx for idx, col in enumerate(list(df.columns))}

        self.dict_attraction2time = {row[dict_col2idx[self.key_attraction]]: row[dict_col2idx[self.key_time]] for row in df.values}
        self.df_master = df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """アトラクション名の正規化"""
        df = make_clean_master(df=df, key_attraction=self.key_attraction)
        return df
