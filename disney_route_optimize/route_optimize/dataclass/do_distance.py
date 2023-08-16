from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from disney_route_optimize.common.config_maneger import ConfigManeger

from ..preprocess.clean_data import make_clean_master

ATTRACTION_NAME = str


@dataclass
class AttractionStepDistance:
    """アトラクション間の歩数を保存するデータクラス
    アトラクション名は正規化したものを使用
    """

    config_maneger: ConfigManeger
    target_attraction: list[ATTRACTION_NAME]
    df_distance: pd.DataFrame = field(default_factory=pd.DataFrame)
    dict_col2idx: dict[ATTRACTION_NAME, int] = field(default_factory=dict)
    key_from: str = "from_attraction"
    key_to: str = "to_attraction"

    def __post_init__(self):
        df_distance = pd.read_csv(Path(self.config_maneger.config.input.path_distance_file))
        df = self._clean_data(df=df_distance)
        df = df.loc[df[self.key_from].isin(self.target_attraction)]
        self.df_distance = df.loc[df[self.key_to].isin(self.target_attraction)]
        self.dict_col2idx = {col: idx for idx, col in enumerate(list(self.df_distance.columns))}

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """アトラクション名の正規化"""
        df = make_clean_master(df=df, key_attraction=self.key_from)
        df = make_clean_master(df=df, key_attraction=self.key_to)

        return df
