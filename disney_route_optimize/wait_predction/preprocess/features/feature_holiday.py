from datetime import datetime, timedelta

import jpholiday
import pandas as pd


def make_dict_isholiday_isweekend(list_date: list[datetime]) -> dict[datetime, dict[str, bool]]:
    """データ内の全日付に対して祝日と週末の判定を行った辞書を作成"""
    list_date = sorted(list_date)
    dict_isholiday_isweekend = dict()
    first_date = list_date[0]
    last_date = list_date[-1]
    list_date = pd.date_range(
        start=first_date - timedelta(days=1), end=last_date + timedelta(days=1), freq="D"
    )
    for date in list_date:
        is_holiday = jpholiday.is_holiday(date)
        is_weekend = 5 <= date.day_of_week  # 土日の場合
        dict_isholiday_isweekend[date] = {"is_holiday": is_holiday, "is_weekend": is_weekend}
    return dict_isholiday_isweekend


def make_is_dayoff_target(
    target_date: datetime, dict_isholiday_isweekend: dict[datetime, dict[str, bool]]
):
    is_holiday_bef_target, is_weekend_bef_target = dict_isholiday_isweekend[
        target_date - timedelta(days=1)
    ].values()
    is_holiday_aft_target, is_weekend_aft_target = dict_isholiday_isweekend[
        target_date + timedelta(days=1)
    ].values()
    is_holiday_target, is_weekend_target = dict_isholiday_isweekend[
        target_date + timedelta(days=1)
    ].values()
    is_dayoff_bef = is_holiday_bef_target and is_weekend_bef_target
    is_dayoff_aft = is_holiday_aft_target and is_weekend_aft_target
    is_dayoff = is_holiday_target and is_weekend_target
    return is_dayoff_bef, is_dayoff, is_dayoff_aft


def make_dayoff_features(list_date: list[datetime], predict_day: int):
    list_date = sorted(list_date)
    last_date = list_date[-1]
    for i_day in range(1, predict_day + 1):
        list_date.append(last_date + timedelta(days=i_day))
    dict_isholiday_isweekend = make_dict_isholiday_isweekend(list_date)
    dict_dayoff_feature = dict()
    for target_date in list_date[:-predict_day]:
        target_date = pd.to_datetime(target_date)
        is_dayoff_bef, is_dayoff, is_dayoff_aft = make_is_dayoff_target(
            target_date=target_date + timedelta(days=predict_day),
            dict_isholiday_isweekend=dict_isholiday_isweekend,
        )
        dict_dayoff_feature[target_date] = {
            "featbool_is_dayoff_bef": is_dayoff_bef,
            "featbool_is_dayoff": is_dayoff,
            "featbool_is_dayoff_aft": is_dayoff_aft,
            "featbool_is_cont_dayoff": (is_dayoff and is_dayoff_bef)
            or (is_dayoff and is_dayoff_aft),
            "featbool_is_between_dayoff": is_dayoff_bef and is_dayoff_aft,
        }

    return dict_dayoff_feature
