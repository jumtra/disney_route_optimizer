import math

POLE_RADIUS = 6356752.314245  # 極半径
EQUATOR_RADIUS = 6378137.0  # 赤道半径

LATITUDE = float
LOGNITUDE = float


# https://www.gis-py.com/entry/py-latlon2distance#google_vignette


def calc_distance(from_pos: tuple[LATITUDE, LOGNITUDE], to_pos: tuple[LATITUDE, LOGNITUDE]):
    """ヒュベニの公式を用いた座標間距離の計算"""
    from_lat, from_log = from_pos
    to_lat, to_log = to_pos

    # 緯度経度をラジアンに変換
    lat_from = math.radians(from_lat)
    lon_from = math.radians(from_log)
    lat_to = math.radians(to_lat)
    lon_to = math.radians(to_log)

    lat_difference = lat_from - lat_to  # 緯度差
    lon_difference = lon_from - lon_to  # 経度差
    lat_average = (lat_from + lat_to) / 2  # 平均緯度

    e2 = (math.pow(EQUATOR_RADIUS, 2) - math.pow(POLE_RADIUS, 2)) / math.pow(
        EQUATOR_RADIUS, 2
    )  # 第一離心率^2

    w = math.sqrt(1 - e2 * math.pow(math.sin(lat_average), 2))

    m = EQUATOR_RADIUS * (1 - e2) / math.pow(w, 3)  # 子午線曲率半径

    n = EQUATOR_RADIUS / w  # 卯酉線曲半径

    distance = math.sqrt(
        math.pow(m * lat_difference, 2) + math.pow(n * lon_difference * math.cos(lat_average), 2)
    )  # 距離計測

    return distance
