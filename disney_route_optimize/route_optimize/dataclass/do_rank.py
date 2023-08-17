from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from disney_route_optimize.common.config_maneger import ConfigManeger

from ..preprocess.clean_data import make_clean_master

ATTRACTION_NAME = str
RANK = float


@dataclass
class AttractionRank:
    """アトラクション別の座標を保存するデータクラス
    アトラクション名は正規化したものを使用
    """

    config_maneger: ConfigManeger
    dict_attraction2rank: dict[ATTRACTION_NAME, RANK] = field(default_factory=dict)
    list_first_rank: list[ATTRACTION_NAME] = field(default_factory=list)
    list_second_rank: list[ATTRACTION_NAME] = field(default_factory=list)
    key_attraction: str = "attraction"
    key_rank: str = "rank"

    def __post_init__(self):
        split_rank: int = self.config_maneger.config.rank.split_rank
        path_rank = Path(self.config_maneger.config.input.path_rank_file)
        df_rank = pd.read_csv(path_rank)
        df = self.clean_data(df_rank)
        df = df_rank.sort_values("rank").reset_index(drop=True)
        # 順位がない場合は名前順で順位をつける
        df[self.key_rank] = np.arange(1, df_rank.shape[0] + 1)
        dict_col2idx = {col: idx for idx, col in enumerate(list(df.columns))}

        self.dict_attraction2rank = {row[dict_col2idx[self.key_attraction]]: row[dict_col2idx[self.key_rank]] for row in df.values}

        if split_rank >= df[self.key_rank].max():
            self.list_first_rank = sorted(df[self.key_attraction].to_list())
            self.list_second_rank = sorted(df[self.key_attraction].to_list())
        else:
            self.list_first_rank = sorted(df.loc[df[self.key_rank] <= split_rank][self.key_attraction].to_list())
            self.list_second_rank = sorted(df.loc[df[self.key_rank] > split_rank][self.key_attraction].to_list())
        self.df_rank = df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """アトラクション名の正規化"""
        df = make_clean_master(df=df, key_attraction=self.key_attraction)
        return df
