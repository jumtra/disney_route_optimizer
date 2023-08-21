from argparse import ArgumentParser


def common_parser() -> ArgumentParser:
    """src/main.pyで使用するargparser"""
    parser = ArgumentParser(description="src/main.pyで使用するargparser")
    parser.add_argument("--load_from", type=str, default=None, help="学習済みモデルをloadする場合は指定：result直下のディレクトリを指定（result/hogehoge）")
    parser.add_argument("--land_type", type=str, default=None, help="対象パークを選択：tdl(ディズニーランド) or tds(ディズニーシー) or both")
    parser.add_argument("--predict_date", type=str, default=None, help="予測実施日：yyyy-mm-ddで指定")
    parser.add_argument("--visit_date", type=str, default=None, help="来園予定日：yyyy-mm-ddで指定")
    parser.add_argument("--visit_time", type=str, default=None, help="入園時刻：hh:mmで指定")
    parser.add_argument("--return_time", type=str, default=None, help="退園時刻：hh:mmで指定")
    parser.add_argument("--cpu_level", type=str, default=None, help="PC使用率の制御：high(全コア), middle（3/4のコア）, low(半分のコア)")

    return parser.parse_args()
