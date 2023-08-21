from datetime import datetime

import numpy as np
import pandas as pd


def make_dict_weather(df_weather: pd.DataFrame) -> dict[datetime, pd.DataFrame]:
    """日付に対する天気データの辞書を作成"""
    dict_weather = {date: df for date, df in df_weather.groupby("date")}
    return dict_weather


def make_dict_weather_features(df_weather: pd.DataFrame) -> dict[datetime, list[float]]:
    df_weather = df_weather.set_index("date")  # [key_use_featuers]
    df = df_weather.copy()
    for i in [2, 3, 4, 5, 6]:
        df.merge(
            df_weather.rolling(i).max(numeric_only=True).add_suffix(f"_roll{i}_max"),
            right_index=True,
            left_index=True,
        )
        df.merge(
            df_weather.rolling(i).min(numeric_only=True).add_suffix(f"_roll{i}_min"),
            right_index=True,
            left_index=True,
        )
        df.merge(
            df_weather.rolling(i).mean(numeric_only=True).add_suffix(f"_roll{i}_mean"),
            right_index=True,
            left_index=True,
        )
        df.merge(
            df_weather.rolling(i).var(numeric_only=True).add_suffix(f"_roll{i}_var"),
            right_index=True,
            left_index=True,
        )
    dict_weather_features = df.add_prefix("featnum_").to_dict(orient="index")
    return dict_weather_features
