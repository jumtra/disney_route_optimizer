from datetime import datetime
from logging import getLogger
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from disney_route_optimize.common.config_maneger import ConfigManeger

logger = getLogger(__name__)


def detect_annomaly(
    df_waittime: pd.DataFrame,
    th_annomaly: float = 60,
    key_date: str = "date",
    key_attraction: str = "attraction",
    key_waittime: str = "wait_time",
) -> pd.DataFrame:
    df_return = df_waittime.copy()
    dict_date2df = {date: df for date, df in df_waittime.groupby(key_date)}
    for _, df in tqdm(dict_date2df.items(), desc="異常検知"):
        for _, df_attraction in df.groupby(key_attraction):
            remove_index = df_attraction[df_attraction[key_waittime].diff() >= th_annomaly].index
            if len(remove_index) > 0:
                logger.warning("異常値が検出されました")
                df_return = df_return.drop(index=remove_index)
    return df_return.reset_index(drop=True)


def interpolate_na(
    df_waittime: pd.DataFrame,
    key_date: str = "date",
    key_attraction: str = "attraction",
    key_waittime: str = "wait_time",
) -> pd.DataFrame:
    df_return = df_waittime.copy()
    dict_date2df = {date: df for date, df in df_waittime.groupby(key_date)}
    for _, df in tqdm(dict_date2df.items(), desc="欠損値補完"):
        for _, df_attraction in df.groupby(key_attraction):
            if not all(np.isnan(df_attraction[[key_waittime]].values)):
                df_temp = df_attraction[[key_waittime]].interpolate(
                    method="linear", limit_direction="both"
                )
                index = df_temp.index
                ser_waittime = df_temp[key_waittime].values
                df_return.loc[index, key_waittime] = ser_waittime
    return df_return.reset_index(drop=True)


def make_clean_wait(
    path_dir: Path, start_date: datetime, end_date: datetime, config_maneger: ConfigManeger
) -> pd.DataFrame:
    """待ち時間データのクリーニングする関数"""
    logger.info("待ち時間データの前処理")
    list_df = []
    assert start_date < end_date
    th_annomary = config_maneger.config.preprocess.th_annomaly
    max_value = config_maneger.config.preprocess.max_value
    min_value = config_maneger.config.preprocess.min_value
    for csv_file in tqdm(sorted(path_dir.glob("**/*.csv")), desc="make_cleaned_waittime_data"):
        # TODO panderaを通す
        date_str = csv_file.parts[-1].split(".")[0].split("_")[0]
        date = datetime.strptime(date_str, "%Y-%m-%d")
        if date < start_date or end_date < date:
            continue
        df_read = pd.read_csv(csv_file, index_col=0).reset_index()
        df_read["date"] = pd.to_datetime(df_read["date"])
        list_df.append(df_read)

    df_waittime = pd.concat(list_df).melt(
        id_vars=["index", "date", "weekday", "time"], var_name="attraction", value_name="wait_time"
    )
    df_waittime["day"] = df_waittime["date"].dt.day
    df_waittime["month"] = df_waittime["date"].dt.month
    df_waittime["year"] = df_waittime["date"].dt.year
    df_waittime["wait_time"] = (
        df_waittime["wait_time"]
        .replace({"－": np.nan, "一時運休": np.nan, "計画運休": np.nan, "案内終了": np.nan})
        .astype(float)
    )
    #
    df_waittime = df_waittime.rename(columns={"index": "num_time"})

    # アトラクション名を正規化
    df_waittime["attraction"] = df_waittime["attraction"].replace(DICT_RENAME)

    # 異常値削除（異常増加と異常減少）
    df_waittime = detect_annomaly(df_waittime=df_waittime, th_annomaly=th_annomary)

    # 上限と下限置き換え
    df_waittime["wait_time"] = df_waittime["wait_time"].mask(
        df_waittime["wait_time"] > max_value, np.nan
    )
    df_waittime["wait_time"] = df_waittime["wait_time"].mask(
        df_waittime["wait_time"] < min_value, 0
    )

    # 欠損値補間
    df_waittime = interpolate_na(df_waittime=df_waittime)
    return df_waittime


def _aggregate_df(df: pd.DataFrame) -> pd.DataFrame:
    # dateとtimeのカラムを除外したデータフレームを作成
    df_processed = df.copy().drop(columns=["date", "time"])

    # 集計結果を格納するデータフレームを初期化
    aggregate_df = pd.DataFrame(
        columns=["date"]
        + [col + "_max" for col in df_processed.columns]
        + [col + "_min" for col in df_processed.columns]
        + [col + "_mean" for col in df_processed.columns]
        + [col + "_var" for col in df_processed.columns]
    )

    # 最大値、最小値、平均値、分散を計算して集計結果を追加
    aggregate_df.loc[0] = (
        [df.at[0, "date"]]
        + df_processed.max().tolist()
        + df_processed.min().tolist()
        + df_processed.mean().tolist()
        + df_processed.var().tolist()
    )

    # 集計結果のデータフレームを表示
    return aggregate_df


def make_clean_weather(path_dir: Path, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """天気データのクリーニングする関数"""
    logger.info("天気データの前処理")
    list_df = []
    assert start_date < end_date
    for csv_file in tqdm(sorted(path_dir.glob("**/*.csv")), desc="make_cleaned_weather_data"):
        date_str = csv_file.parts[-1].split(".")[0]
        date = datetime.strptime(date_str, "%Y-%m-%d")
        if date < start_date or end_date < date:
            continue
        # TODO panderaを通す
        df_read = pd.read_csv(csv_file)
        df_read["date"] = pd.to_datetime(df_read["date"])
        df_read = _aggregate_df(df_read)
        list_df.append(df_read)

    df_weather = pd.concat(list_df)
    return df_weather


def make_clean_data(config_maneger: ConfigManeger):
    start_date = datetime.strptime(config_maneger.config.common.train_start_date, "%Y-%m-%d")
    end_date = datetime.strptime(config_maneger.config.common.predict_date, "%Y-%m-%d")
    df_weather = make_clean_weather(
        Path(config_maneger.config.input.path_weather_dir), start_date, end_date
    )

    if config_maneger.config.common.land_type == "both":
        config_maneger.config.common.land_type = "tdl"
        df_waittime_tdl = make_clean_wait(
            Path(config_maneger.config.input.path_waittime_dir),
            start_date,
            end_date,
            config_maneger,
        )
        config_maneger.config.common.land_type = "tds"
        df_waitime_tds = make_clean_wait(
            Path(config_maneger.config.input.path_waittime_dir),
            start_date,
            end_date,
            config_maneger,
        )
        df_waittime = pd.concat([df_waittime_tdl, df_waitime_tds], axis=0).reset_index(drop=True)
    else:
        df_waittime = make_clean_wait(
            Path(config_maneger.config.input.path_waittime_dir),
            start_date,
            end_date,
            config_maneger,
        )

    # df_master = make_clean_master(Path(config_maneger.config.input.path_master_file))

    return df_weather, df_waittime
