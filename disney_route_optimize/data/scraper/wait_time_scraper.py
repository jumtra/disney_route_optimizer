import datetime
import time
from logging import getLogger
from pathlib import Path

import pandas as pd

from disney_route_optimize.data.scraper.common_scraper import get_text_from_url

logger = getLogger(__name__)


def _get_attraction_name(list_table):
    """tableからアトラクション名を取得する"""
    list_title = ["time"]
    for _title in list_table.find_all("th", class_="t_h"):
        title = _title.get_text().replace("\n", "")
        list_title.append(str(title))
    return list_title


def _get_wait_time(list_table):
    """tableから待ち時間を取得する"""
    list_row = []
    for row in list_table.find_all("tr", class_=f"t_cool_b date_width"):
        _row = row.get_text(separator=",", strip=True).split(",")
        list_row.append(_row)
    return list_row


def _get_not_exist_date(_date: datetime, dir_data: Path) -> datetime:
    """データが存在しない起点を捜索する関数"""
    _year = _date.year
    _month = str(_date.month).zfill(2)
    _day = str(_date.day).zfill(2)
    is_exist = Path(dir_data / f"{_year}-{_month}-{_day}_waittime.csv").exists()
    while is_exist:
        logger.warning(f"{_year}-{_month}-{_day}のデータは既にあります。")
        _date += datetime.timedelta(days=1)
        _year = _date.year
        _month = str(_date.month).zfill(2)
        _day = str(_date.day).zfill(2)
        is_exist_next1 = Path(dir_data / f"{_year}-{_month}-{_day}_waittime.csv").exists()

        _date_next = _date + datetime.timedelta(days=1)
        __year = _date_next.year
        __month = str(_date_next.month).zfill(2)
        __day = str(_date_next.day).zfill(2)
        is_exist_next2 = Path(dir_data / f"{__year}-{__month}-{__day}_waittime.csv").exists()

        _date_next = _date + datetime.timedelta(days=2)
        __year = _date_next.year
        __month = str(_date_next.month).zfill(2)
        __day = str(_date_next.day).zfill(2)
        is_exist_next3 = Path(dir_data / f"{__year}-{__month}-{__day}_waittime.csv").exists()

        is_exist = is_exist_next1 or is_exist_next2 or is_exist_next3

    return _date


def get_wait_time_data(dir_data: Path, is_land=True):
    # ディニーランドの場合
    if is_land:
        base_url = """https://urtrip.jp/tdl-past-info/?rm={_year}{_month}{_day}"""
        dir_data = dir_data / "tdl/wait_time"
    # ディニーシーの場合
    else:
        base_url = """https://urtrip.jp/tds-past-info/?rm={_year}{_month}{_day}"""
        dir_data = dir_data / "tds/wait_time"

    # 親ディレクトリ含めて作成
    dir_data.mkdir(parents=True, exist_ok=True)
    # 2022/4/25以降のデータしかないのでそこを起点に取得
    _date = datetime.date(year=2022, month=4, day=25)
    _date = _get_not_exist_date(_date=_date, dir_data=dir_data)

    # 実行日を取得
    _to_day = datetime.date.today()

    while _date < _to_day:
        _year = _date.year
        _month = str(_date.month).zfill(2)
        _day = str(_date.day).zfill(2)
        # URL指定
        url = base_url.format(_year=_year, _month=_month, _day=_day)
        soup = get_text_from_url(url=url)
        list_table = soup.find_all("div", class_="t_area")
        try:
            # アトラクションからグリーティングのテーブル情報を引っ張る
            for i_table in range(5):
                list_row = _get_wait_time(list_table=list_table[i_table])
                list_title = _get_attraction_name(list_table=list_table[i_table])
                if i_table == 0:
                    df = pd.DataFrame(list_row, columns=list_title)
                else:
                    df = df.merge(pd.DataFrame(list_row, columns=list_title), on="time")
            df["date"] = _date
            df["weekday"] = _date.weekday()
            # カラムの正規化
            df.columns = df.columns.str.replace("”", "").str.replace("“", "").str.replace("\r", "")
            logger.info(f"{_year}-{_month}-{_day}のデータを取得しました。")
            df.to_csv(dir_data / f"{_year}-{_month}-{_day}_waittime.csv")
        except Exception as e:
            logger.error(f"データの取得に失敗しました。\n{e}")
        _date += datetime.timedelta(days=1)
        time.sleep(1)


if __name__ == "__main__":
    get_wait_time_data(dir_data=Path("data"), is_land=False)
