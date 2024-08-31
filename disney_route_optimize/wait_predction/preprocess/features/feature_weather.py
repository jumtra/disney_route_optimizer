from datetime import datetime

import pandas as pd


def make_dict_weather(df_weather: pd.DataFrame) -> dict[datetime, pd.DataFrame]:
    """日付に対する天気データの辞書を作成"""
    dict_weather = {date: df for date, df in df_weather.groupby("date")}
    return dict_weather


def make_dict_weather_features(df_weather: pd.DataFrame) -> dict[datetime, list[float]]:
    df_weather = df_weather.set_index("date")  # [key_use_featuers]
    key_use_featuers = [
        "temperature_max",
        "temperature_mean",
        "temperature_min",
        "temperature_var",
        "precipitation_max",
        "precipitation_mean",
        "precipitation_min",
        "precipitation_var",
        "sunshine_hours_max",
        "sunshine_hours_mean",
        "sunshine_hours_min",
        "sunshine_hours_var",
        "wind_speed_max",
        "wind_speed_mean",
        "wind_speed_min",
        "wind_speed_var",
    ]
    df = df_weather[key_use_featuers].copy()
    # NOTE add this code is memory error
    # for i in [7]:
    #    df = df.merge(df_weather.rolling(i).max(numeric_only=True).add_suffix(f"_roll{i}_max"), right_index=True, left_index=True, how="left")
    #    df = df.merge(df_weather.rolling(i).min(numeric_only=True).add_suffix(f"_roll{i}_min"), right_index=True, left_index=True, how="left")
    #    df = df.merge(df_weather.rolling(i).mean(numeric_only=True).add_suffix(f"_roll{i}_mean"), right_index=True, left_index=True, how="left")
    #    df = df.merge(df_weather.rolling(i).var(numeric_only=True).add_suffix(f"_roll{i}_var"), right_index=True, left_index=True, how="left")
    dict_weather_features = df.add_prefix("featnum_").to_dict(orient="index")
    return dict_weather_features
