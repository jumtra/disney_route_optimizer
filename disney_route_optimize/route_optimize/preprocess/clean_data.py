import pandas as pd

from .rename_attraction_name import DICT_RENAME


def make_clean_master(df: pd.DataFrame, key_attraction: str) -> pd.DataFrame:
    """attraction_masterデータのクリーニングする関数"""
    # アトラクション名を正規化
    df[key_attraction] = df[key_attraction].replace(DICT_RENAME)
    return df
