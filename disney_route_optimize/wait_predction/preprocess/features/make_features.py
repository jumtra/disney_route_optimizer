import gc
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from disney_route_optimize.common.config_manager import ConfigManager

from .feature_cluster import make_feat_clsuter
from .feature_holiday import make_dayoff_features
from .feature_waittime import make_waittime_featuers
from .feature_weather import make_dict_weather_features


def make_now_waittime_features(dict_numtime2target: dict[int, float], num_time: int, is_test: bool, recently_num: int = 5):
    list_name = []
    list_feat = []

    if is_test:
        list_recently = [np.nan for _ in range(recently_num)]
    else:
        list_recently_target = [value for i, value in enumerate(list(dict_numtime2target.values())) if i < num_time]
        if len(list_recently_target) < recently_num:
            missing_num = recently_num - len(list_recently_target)
            list_recently = [np.nan for _ in range(missing_num)] + list_recently_target
        else:
            list_recently = list_recently_target[len(list_recently_target) - recently_num :]

    list_feat.extend(list_recently)
    list_name.extend([f"featnum_recently_pre{i}" for i in range(recently_num)])

    list_feat.append(np.nanmean(list_recently))
    list_name.append(f"featnum_recently{recently_num}_mean")

    list_feat.append(np.nanmin(list_recently))
    list_name.append(f"featnum_recently{recently_num}_min")

    list_feat.append(np.nanmax(list_recently))
    list_name.append(f"featnum_recently{recently_num}_max")

    list_feat.append(np.nanstd(list_recently))
    list_name.append(f"featnum_recently{recently_num}_std")

    return list_feat, list_name


def make_dict_target(
    df_waittime: pd.DataFrame,
    key_attraction: str = "attraction",
    key_numtime: str = "num_time",
    key_waittime: str = "wait_time",
    key_date: str = "date",
):
    dict_target = {}
    for col_attr, df_attr in df_waittime.groupby([key_attraction]):
        dict_date = {}
        for col_date, df_date in df_attr.groupby([key_date]):
            dict_date[col_date] = df_date[[key_numtime, key_waittime]].set_index(key_numtime).to_dict()[key_waittime]
        dict_target[col_attr] = dict_date
    return dict_target


def make_features(
    df_waittime: pd.DataFrame,
    df_weather: pd.DataFrame,
    predict_start_date: datetime,
    config_manager: ConfigManager,
    key_attraction: str = "attraction",
    key_numtime: str = "num_time",
    key_waittime: str = "wait_time",
    key_date: str = "date",
    key_cluster: str = "cluster",
) -> pd.DataFrame:
    recently_num = config_manager.config.feature.recently_num
    n_jobs = config_manager.config.feature.n_jobs
    predict_day = config_manager.config.predict.predict_day
    dict_feat_attraction2cluster = make_feat_clsuter(df=df_waittime, config_manager=config_manager)
    df_waittime[key_date] = pd.to_datetime(df_waittime[key_date])
    df_weather[key_date] = pd.to_datetime(df_weather[key_date])
    list_date = list(df_waittime[key_date].dt.date.unique())
    list_date = sorted(list_date, reverse=True)

    dict_target = make_dict_target(df_waittime=df_waittime)
    dict_dayoff_features = make_dayoff_features(list_date, predict_day)
    dict_weather_featuers = make_dict_weather_features(df_weather=df_weather)
    dict_feat_attraction2int = {attraction: attr_i for attr_i, attraction in enumerate(df_waittime[key_attraction].unique())}

    def process_attraction(attraction_name, df_attraction):
        feature_columns = [
            "attraction_name",
            "date",
            "cluster",
            "featcat_cluster",
            "featcat_attraction",
            "featcat_year",
            "featcat_month",
            "featcat_day",
            "featcat_weekday",
            "featcat_numtime",
            "featnum_global_max",
            "featnum_global_min",
            "featnum_global_mean",
            "featnum_global_median",
            "featnum_global_var",
            "featnum_global_std",
        ]
        is_get_col = False
        dict_attr2agg = df_attraction[[key_waittime]].aggregate({"max", "min", "mean", "median", "var", "std"}).to_dict()[key_waittime]
        dict_date2df = {date: df for date, df in df_attraction.groupby(key_date)}
        dict_numtime2df = {
            num_time: df[[key_date, key_waittime]].set_index(key_date).sort_index() for num_time, df in df_attraction.groupby(key_numtime)
        }
        dict_waittime_features = make_waittime_featuers(list_date, dict_numtime2df, predict_day)
        dict_target_value = dict_target[attraction_name]

        list_feat = []
        list_test = []
        for target_date, df_y in sorted(dict_date2df.items(), reverse=True):
            weekday = (target_date + timedelta(days=predict_day)).weekday()
            year = (target_date + timedelta(days=predict_day)).year
            month = (target_date + timedelta(days=predict_day)).month
            day = (target_date + timedelta(days=predict_day)).day

            for row in df_y.to_dict(orient="index").values():
                num_time = row[key_numtime]
                cluster = row[key_cluster]
                dict_numtime2target = dict_target_value.get(pd.to_datetime(target_date) + timedelta(days=predict_day), None)
                if dict_numtime2target is None:
                    wait_time = np.nan
                else:
                    wait_time = dict_numtime2target.get(int(num_time), np.nan)

                if np.isnan(wait_time) and ((pd.to_datetime(target_date) + timedelta(days=predict_day)) <= pd.to_datetime(predict_start_date)):
                    continue

                list_x = [
                    attraction_name,
                    target_date,
                    cluster,
                    dict_feat_attraction2cluster[attraction_name],
                    dict_feat_attraction2int[attraction_name],
                    year,
                    month,
                    day,
                    weekday,
                    num_time,
                    dict_attr2agg["max"],
                    dict_attr2agg["min"],
                    dict_attr2agg["mean"],
                    dict_attr2agg["median"],
                    dict_attr2agg["var"],
                    dict_attr2agg["std"],
                ]
                feat_waittime = dict_waittime_features[target_date.date()][num_time]

                is_test = predict_start_date.date() < (pd.to_datetime(target_date) + timedelta(days=predict_day)).date()
                feat_recent, feat_name = make_now_waittime_features(
                    dict_numtime2target,
                    num_time=num_time,
                    is_test=is_test,
                    recently_num=recently_num,
                )
                list_x.extend(feat_recent)
                if not is_get_col:
                    feature_columns.extend(feat_name)

                if not is_get_col:
                    feature_columns.extend(list(feat_waittime.keys()))

                feat_dayoff = dict_dayoff_features[pd.to_datetime(target_date)]
                if not is_get_col:
                    feature_columns.extend(list(feat_dayoff.keys()))

                feat_weather = dict_weather_featuers[pd.to_datetime(target_date)]
                if not is_get_col:
                    feature_columns.extend(list(feat_weather.keys()))
                    feature_columns.append("target")
                    is_get_col = True

                if all(np.isnan(list(feat_weather.values()))):
                    continue

                if sum(np.isnan(list(feat_waittime.values()))) >= int(len(list(feat_waittime.values())) * 0.9):
                    continue

                list_x.extend(list(feat_waittime.values()))
                list_x.extend(list(feat_dayoff.values()))
                list_x.extend(list(feat_weather.values()))
                list_x.append(wait_time)

                if is_test:
                    list_test.append(list_x)
                else:
                    list_feat.append(list_x)
        return list_feat, list_test, feature_columns

    result = Parallel(n_jobs=int(n_jobs / 2))(
        delayed(process_attraction)(attraction_name, df_attraction)
        for attraction_name, df_attraction in tqdm(df_waittime.groupby(key_attraction), desc="make_features")
    )
    list_feat = []
    list_test = []
    for feat, test, _ in result:
        list_feat.extend(feat)
        list_test.extend(test)
    feature_columns = result[0][2]
    del (
        result,
        dict_target,
        dict_dayoff_features,
        dict_feat_attraction2cluster,
        dict_weather_featuers,
    )
    gc.collect()
    df = pd.DataFrame(list_feat, columns=feature_columns)
    del list_feat
    gc.collect()

    df_test = pd.DataFrame(list_test, columns=feature_columns)
    del list_test
    gc.collect()
    return df, df_test
