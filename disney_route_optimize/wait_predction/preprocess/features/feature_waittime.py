import warnings
from datetime import datetime, timedelta
from logging import getLogger

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
logger = getLogger(__name__)


def _get_feat(dict_numtime2feat: dict, numtime: int, date: datetime, feat: str):
    try:
        if "weekday" in feat:
            return dict_numtime2feat[numtime][feat][date.weekday()][pd.to_datetime(date)]

        return dict_numtime2feat[numtime][feat][pd.to_datetime(date)]
    except Exception as e:
        # logger.error(f"{feat}:{date}のデータが存在しません")
        return np.nan


def make_waittime_featuers(list_date: list[datetime], dict_numtime2df: dict[int, pd.DataFrame], predict_day: int) -> list:
    dict_waittime_features = dict()

    dict_numtime2feat = dict()

    for numtime, df in dict_numtime2df.items():
        dict_feat_mean_week1 = df.rolling(7).mean().to_dict()["wait_time"]
        dict_feat_mean_week2 = df.rolling(14).mean().to_dict()["wait_time"]
        dict_feat_mean_week3 = df.rolling(21).mean().to_dict()["wait_time"]
        dict_feat_mean_week4 = df.rolling(28).mean().to_dict()["wait_time"]
        dict_feat_median_week1 = df.rolling(7).median().to_dict()["wait_time"]
        dict_feat_median_week2 = df.rolling(14).median().to_dict()["wait_time"]
        dict_feat_median_week3 = df.rolling(21).median().to_dict()["wait_time"]
        dict_feat_median_week4 = df.rolling(28).median().to_dict()["wait_time"]
        dict_feat_max_week1 = df.rolling(7).max().to_dict()["wait_time"]
        dict_feat_max_week2 = df.rolling(14).max().to_dict()["wait_time"]
        dict_feat_max_week3 = df.rolling(21).max().to_dict()["wait_time"]
        dict_feat_max_week4 = df.rolling(28).max().to_dict()["wait_time"]
        dict_feat_min_week1 = df.rolling(7).min().to_dict()["wait_time"]
        dict_feat_min_week2 = df.rolling(14).min().to_dict()["wait_time"]
        dict_feat_min_week3 = df.rolling(21).min().to_dict()["wait_time"]
        dict_feat_min_week4 = df.rolling(28).min().to_dict()["wait_time"]
        dict_feat_diff_day1 = df.diff(1).to_dict()["wait_time"]
        dict_feat_diff_day2 = df.diff(2).to_dict()["wait_time"]
        dict_feat_diff_day3 = df.diff(3).to_dict()["wait_time"]

        df_weekdays = df.copy()
        df_weekdays["weekday"] = df.index.weekday

        dict_feat_weekday_shift1 = {weekday: df_weekday.shift(1).to_dict()["wait_time"] for weekday, df_weekday in df_weekdays.groupby("weekday")}
        dict_feat_weekday_shift2 = {weekday: df_weekday.shift(2).to_dict()["wait_time"] for weekday, df_weekday in df_weekdays.groupby("weekday")}
        dict_feat_weekday_shift3 = {weekday: df_weekday.shift(3).to_dict()["wait_time"] for weekday, df_weekday in df_weekdays.groupby("weekday")}
        dict_feat_weekday_shift4 = {weekday: df_weekday.shift(4).to_dict()["wait_time"] for weekday, df_weekday in df_weekdays.groupby("weekday")}
        dict_feat_weekday_diff1 = {weekday: df_weekday.diff(1).to_dict()["wait_time"] for weekday, df_weekday in df_weekdays.groupby("weekday")}
        dict_feat_weekday_diff2 = {weekday: df_weekday.diff(2).to_dict()["wait_time"] for weekday, df_weekday in df_weekdays.groupby("weekday")}
        dict_feat_weekday_diff3 = {weekday: df_weekday.diff(3).to_dict()["wait_time"] for weekday, df_weekday in df_weekdays.groupby("weekday")}
        dict_feat_weekday_diff4 = {weekday: df_weekday.diff(4).to_dict()["wait_time"] for weekday, df_weekday in df_weekdays.groupby("weekday")}

        dict_feat_weekday_mean_roll2 = {
            weekday: df_weekday.rolling(2).mean().to_dict()["wait_time"] for weekday, df_weekday in df_weekdays.groupby("weekday")
        }
        dict_feat_weekday_mean_roll3 = {
            weekday: df_weekday.rolling(3).mean().to_dict()["wait_time"] for weekday, df_weekday in df_weekdays.groupby("weekday")
        }
        dict_feat_weekday_mean_roll4 = {
            weekday: df_weekday.rolling(4).mean().to_dict()["wait_time"] for weekday, df_weekday in df_weekdays.groupby("weekday")
        }
        dict_feat_weekday_median_roll2 = {
            weekday: df_weekday.rolling(2).median().to_dict()["wait_time"] for weekday, df_weekday in df_weekdays.groupby("weekday")
        }
        dict_feat_weekday_median_roll3 = {
            weekday: df_weekday.rolling(3).median().to_dict()["wait_time"] for weekday, df_weekday in df_weekdays.groupby("weekday")
        }
        dict_feat_weekday_median_roll4 = {
            weekday: df_weekday.rolling(4).median().to_dict()["wait_time"] for weekday, df_weekday in df_weekdays.groupby("weekday")
        }

        dict_feat_weekday_max_roll2 = {
            weekday: df_weekday.rolling(2).max().to_dict()["wait_time"] for weekday, df_weekday in df_weekdays.groupby("weekday")
        }
        dict_feat_weekday_max_roll3 = {
            weekday: df_weekday.rolling(3).max().to_dict()["wait_time"] for weekday, df_weekday in df_weekdays.groupby("weekday")
        }
        dict_feat_weekday_max_roll4 = {
            weekday: df_weekday.rolling(4).max().to_dict()["wait_time"] for weekday, df_weekday in df_weekdays.groupby("weekday")
        }
        dict_feat_weekday_min_roll2 = {
            weekday: df_weekday.rolling(2).min().to_dict()["wait_time"] for weekday, df_weekday in df_weekdays.groupby("weekday")
        }
        dict_feat_weekday_min_roll3 = {
            weekday: df_weekday.rolling(3).min().to_dict()["wait_time"] for weekday, df_weekday in df_weekdays.groupby("weekday")
        }
        dict_feat_weekday_min_roll4 = {
            weekday: df_weekday.rolling(4).min().to_dict()["wait_time"] for weekday, df_weekday in df_weekdays.groupby("weekday")
        }

        dict_numtime2feat[numtime] = {
            "mean_week1": dict_feat_mean_week1,
            "mean_week2": dict_feat_mean_week2,
            "mean_week3": dict_feat_mean_week3,
            "mean_week4": dict_feat_mean_week4,
            "median_week1": dict_feat_median_week1,
            "median_week2": dict_feat_median_week2,
            "median_week3": dict_feat_median_week3,
            "median_week4": dict_feat_median_week4,
            "max_week1": dict_feat_max_week1,
            "max_week2": dict_feat_max_week2,
            "max_week3": dict_feat_max_week3,
            "max_week4": dict_feat_max_week4,
            "min_week1": dict_feat_min_week1,
            "min_week2": dict_feat_min_week2,
            "min_week3": dict_feat_min_week3,
            "min_week4": dict_feat_min_week4,
            "diff_day1": dict_feat_diff_day1,
            "diff_day2": dict_feat_diff_day2,
            "diff_day3": dict_feat_diff_day3,
            "weekday_shift1": dict_feat_weekday_shift1,
            "weekday_shift2": dict_feat_weekday_shift2,
            "weekday_shift3": dict_feat_weekday_shift3,
            "weekday_shift4": dict_feat_weekday_shift4,
            "weekday_diff1": dict_feat_weekday_diff1,
            "weekday_diff2": dict_feat_weekday_diff2,
            "weekday_diff3": dict_feat_weekday_diff3,
            "weekday_diff4": dict_feat_weekday_diff4,
            "weekday_mean_roll2": dict_feat_weekday_mean_roll2,
            "weekday_mean_roll3": dict_feat_weekday_mean_roll3,
            "weekday_mean_roll4": dict_feat_weekday_mean_roll4,
            "weekday_median_roll2": dict_feat_weekday_median_roll2,
            "weekday_median_roll3": dict_feat_weekday_median_roll3,
            "weekday_median_roll4": dict_feat_weekday_median_roll4,
            "weekday_max_roll2": dict_feat_weekday_max_roll2,
            "weekday_max_roll3": dict_feat_weekday_max_roll3,
            "weekday_max_roll4": dict_feat_weekday_max_roll4,
            "weekday_min_roll2": dict_feat_weekday_min_roll2,
            "weekday_min_roll3": dict_feat_weekday_min_roll3,
            "weekday_min_roll4": dict_feat_weekday_min_roll4,
        }

    for date in list_date:
        dict_featnumtime = dict()
        for numtime, dict_numtime in dict_numtime2feat.items():
            dict_feat = {
                "featnum_time1_shift1": _get_feat(dict_numtime2feat, numtime + 1, date, "weekday_shift1"),
                "featnum_time2_shift1": _get_feat(dict_numtime2feat, numtime + 2, date, "weekday_shift1"),
                "featnum_time-1_shift1": _get_feat(dict_numtime2feat, numtime - 1, date, "weekday_shift1"),
                "featnum_time-2_shift1": _get_feat(dict_numtime2feat, numtime - 2, date, "weekday_shift1"),
                "featnum_time1_shift2": _get_feat(dict_numtime2feat, numtime + 1, date, "weekday_shift2"),
                "featnum_time2_shift2": _get_feat(dict_numtime2feat, numtime + 2, date, "weekday_shift2"),
                "featnum_time-1_shift2": _get_feat(dict_numtime2feat, numtime - 1, date, "weekday_shift2"),
                "featnum_time-2_shift2": _get_feat(dict_numtime2feat, numtime - 2, date, "weekday_shift2"),
                "featnum_time1_mean_roll2": _get_feat(dict_numtime2feat, numtime + 1, date, "weekday_mean_roll2"),
                "featnum_time2_mean_roll2": _get_feat(dict_numtime2feat, numtime + 2, date, "weekday_mean_roll2"),
                "featnum_time-1_mean_roll2": _get_feat(dict_numtime2feat, numtime - 1, date, "weekday_mean_roll2"),
                "featnum_time-2_mean_roll2": _get_feat(dict_numtime2feat, numtime - 2, date, "weekday_mean_roll2"),
                "featnum_time1_mean_roll4": _get_feat(dict_numtime2feat, numtime + 1, date, "weekday_mean_roll4"),
                "featnum_time2_mean_roll4": _get_feat(dict_numtime2feat, numtime + 2, date, "weekday_mean_roll4"),
                "featnum_time-1_mean_roll4": _get_feat(dict_numtime2feat, numtime - 1, date, "weekday_mean_roll4"),
                "featnum_time-2_mean_roll4": _get_feat(dict_numtime2feat, numtime - 2, date, "weekday_mean_roll4"),
                "featnum_time1_diff1": _get_feat(dict_numtime2feat, numtime + 1, date, "weekday_diff1"),
                "featnum_time2_diff1": _get_feat(dict_numtime2feat, numtime + 2, date, "weekday_diff1"),
                "featnum_time-1_diff1": _get_feat(dict_numtime2feat, numtime - 1, date, "weekday_diff1"),
                "featnum_time-2_diff1": _get_feat(dict_numtime2feat, numtime - 2, date, "weekday_diff1"),
            }

            for feat_name, df_feat in dict_numtime.items():
                try:
                    if "weekday" in feat_name:
                        dict_feat["featnum_" + feat_name] = df_feat[(pd.to_datetime(date) + timedelta(days=predict_day)).weekday()][
                            pd.to_datetime(date) + timedelta(days=predict_day) - timedelta(days=7)
                        ]
                    else:
                        dict_feat["featnum_" + feat_name] = df_feat[pd.to_datetime(date)]

                except Exception as e:
                    dict_feat["featnum_" + feat_name] = np.nan
                    # logger.error(f"{feat_name}:{date}のデータが存在しません")
            dict_featnumtime[numtime] = dict_feat
        dict_waittime_features[date] = dict_featnumtime
    return dict_waittime_features
