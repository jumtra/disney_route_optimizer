from pathlib import Path

import folium
import pandas as pd
from folium import DivIcon, Map

from disney_route_optimize.common.config_manager import ConfigManager


def number_div_icon(color: str, number: int) -> DivIcon:
    """folium上でピンに数字を表示するアイコン"""
    icon = DivIcon(
        icon_size=(150, 36),
        icon_anchor=(14, 40),
        html="""<span class="fa-stack" style="font-size: 12pt;">
                    <!-- The icon that will wrap the number -->
                    <span class="fa fa-circle-o fa-stack-2x" style="color: {:s};"></span>
                    <!-- a strong element with the custom content, in this case a number -->
                    <strong class="fa-stack-1x" style="color: {:s};">
                        {:02d}
                    </strong>
                </span>""".format(
            color, color, number
        ),
    )
    return icon


def calculate_center(coordinates: list[tuple[float, float]]) -> tuple[float, float]:
    """座標の中心を取得する"""
    num_coordinates = len(coordinates)
    sum_lat = sum_lon = 0

    for lat, lon in coordinates:
        sum_lat += lat
        sum_lon += lon

    center_lat = sum_lat / num_coordinates
    center_lon = sum_lon / num_coordinates

    return (center_lat, center_lon)


def _preprocess(df_result: pd.DataFrame, df_pos: pd.DataFrame) -> tuple[dict, list[tuple[float, float]]]:
    """foliumで扱うデータ形式に変換"""
    use_columns = list(df_result.columns) + ["lat", "lon"]

    df = df_result.merge(df_pos, how="left", left_on="施設名", right_on="attraction")[use_columns]

    # スタート地点の座標を取得
    set_pos = df_pos.loc[df_pos["attraction"] == "0_スタート地点"][["lat", "lon"]].values[0]

    # スタート地点の座標を挿入
    df.loc[df.shape[0] - 1, ["lat", "lon"]] = set_pos
    df.loc[0, ["lat", "lon"]] = set_pos

    dict_col2idx = {col: idx for idx, col in enumerate(list(df.columns))}

    # ピン上で表示する情報の取得
    locations = []
    for row in df.values:
        dict_pos = {
            "coords": (row[dict_col2idx["lat"]], row[dict_col2idx["lon"]]),
            "info": {
                "施設名": row[dict_col2idx["施設名"]],
                "到着時刻": row[dict_col2idx["到着時刻"]],
                "出発時刻": row[dict_col2idx["出発時刻"]],
                "予測待ち時間": row[dict_col2idx["予測待ち時間"]],
                "移動時間": row[dict_col2idx["移動時間"]],
            },
        }
        locations.append(dict_pos)

    # 座標をリストで取得
    list_coord = [pos["coords"] for pos in locations]

    return locations, list_coord


def get_map(df_result: pd.DataFrame, df_pos: pd.DataFrame) -> Map:
    """最適化結果から地図を取得"""

    # 前処理後のデータを取得
    locations, list_coord = _preprocess(df_result, df_pos)

    center_pos = calculate_center(list_coord)
    # 地図を作成
    m = folium.Map(location=center_pos, zoom_start=15)

    # 線を描くためのパスを作成
    line = folium.PolyLine(locations=list_coord, color="green")

    # 地図に線を追加
    m.add_child(line)

    # マーカーとポップアップを追加
    for num_i, location in enumerate(locations):
        coords = location["coords"]
        info = location["info"]
        popup_content = "<ul>"
        for key, value in info.items():
            popup_content += f"<li><b> {key}:</b> {value}</li> "
        popup_content += "</ul>"

        if num_i in [0, len(locations) - 1]:
            m.add_child(
                folium.Marker(
                    location=coords,
                    popup=folium.Popup(popup_content),
                    icon=folium.Icon(color="red", icon="home"),
                )
            )
        else:
            # 背景を白色で塗りつぶし
            m.add_child(
                folium.Marker(
                    location=coords,
                    popup=folium.Popup(popup_content),
                    icon=folium.Icon(color="white", icon_color="white"),
                )
            )
            # 順番を入れる　先ほどの自作DivIcon
            m.add_child(
                folium.Marker(
                    location=coords,
                    popup=folium.Popup(popup_content),
                    icon=number_div_icon(color="black", number=num_i),
                )
            )
    return m


def opt_visualize(config_manager: ConfigManager, df_result: pd.DataFrame, df_pos: pd.DataFrame) -> None:
    m = get_map(df_result=df_result, df_pos=df_pos)
    land_type = str(config_manager.config.common.land_type)

    path_optimize = Path(config_manager.config.output.opt_output.path_opt_dir) / land_type

    m.save(path_optimize / "map.html")
