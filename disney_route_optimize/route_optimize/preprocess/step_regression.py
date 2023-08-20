import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ..dataclass.do_distance import AttractionStepDistance
from ..dataclass.do_position import AttractionPosition
from ..preprocess.calc_distance import calc_distance

logger = logging.getLogger(__name__)


@dataclass
class RegressionInput:
    """座標間距離から歩数を予測する回帰モデルの入力データクラス"""

    attraction_position: AttractionPosition
    attraction_distance: AttractionStepDistance
    key_actual: str = "steps"  # 実測値
    key_calc_distance: str = "calc"  # 座標間距離

    def __post_init__(self):
        self.df_train = self.make_train_input()
        self.df_pred = self.make_predict_input()

    def make_train_input(self) -> pd.DataFrame:
        """座標間距離から歩数を予測する回帰の学習入力データ"""
        dict_attr2latlon = self.attraction_position.dict_attraction2pos

        dict_col2idx = self.attraction_distance.dict_col2idx
        df_dis = self.attraction_distance.df_distance
        list_result = []
        for row in df_dis.values:
            from_attr = row[dict_col2idx[self.attraction_distance.key_from]]
            to_attr = row[dict_col2idx[self.attraction_distance.key_to]]
            dis = calc_distance(dict_attr2latlon[from_attr], dict_attr2latlon[to_attr])
            if not np.isnan(dis):
                list_result.append([from_attr, to_attr, dis, row[dict_col2idx[self.key_actual]]])

        df_train = pd.DataFrame(list_result).rename(
            columns={
                0: self.attraction_distance.key_from,
                1: self.attraction_distance.key_to,
                2: self.key_calc_distance,
                3: self.key_actual,
            }
        )
        return df_train

    def make_predict_input(self) -> pd.DataFrame:
        """歩数がないアトラクションの歩数を予測対象データフレーム生成"""

        df_distance = self.attraction_distance.df_distance
        dict_attr2latlon = self.attraction_position.dict_attraction2pos
        list_not_target = list(df_distance[self.attraction_distance.key_from].unique())
        list_res = []
        for from_attraction, from_pos in dict_attr2latlon.items():
            list_temp = []
            if from_attraction not in list_not_target:
                for to_attraction, to_pos in dict_attr2latlon.items():
                    if from_attraction != to_attraction:
                        distance = calc_distance(from_pos, to_pos)
                        if not np.isnan(distance):
                            list_temp.append([from_attraction, to_attraction, distance])
                list_res.extend(list_temp)

        df_pred = pd.DataFrame(
            list_res,
            columns=[
                self.attraction_distance.key_from,
                self.attraction_distance.key_to,
                self.key_calc_distance,
            ],
        )
        return df_pred

    def get_train(self):
        """学習の入力データを取得"""
        X = self.df_train[[self.key_calc_distance]]
        y = self.df_train[[self.key_actual]]
        return X, y

    def get_pred(self):
        """予測の入力データを取得"""
        X = self.df_pred[[self.key_calc_distance]]
        return X


@dataclass
class RegressionFitResult:
    """回帰式の情報を保存するデータクラス"""

    coef: float
    intercept: float
    R2: float

    def to_df(self) -> pd.DataFrame:
        dict_df = {"coef": [self.coef], "intercept": [self.intercept], "R2": [self.R2]}
        return pd.DataFrame(dict_df)


@dataclass
class RegressionResult:
    """回帰の結果を保存するデータクラス"""

    df_train: pd.DataFrame
    df_pred: pd.DataFrame
    fit_result: RegressionFitResult
    key_from: str = "from_attraction"
    key_to: str = "to_attraction"

    def __post_init__(self):
        df_steps = pd.concat([self.df_pred, self.df_train])
        df_replaced = df_steps.copy()
        df_replaced[self.key_from], df_replaced[self.key_to] = df_replaced[self.key_to], df_replaced[self.key_from]
        df_step = pd.concat([df_steps, df_replaced]).drop_duplicates([self.key_from, self.key_to]).reset_index(drop=True)
        df_step = self._change_move_attraction(df_step)
        self.df_steps = df_step

    def _change_move_attraction(self, df_step: pd.DataFrame) -> pd.DataFrame:
        """ディズニーシーの移動アトラクションを移動コストに反映"""
        move_attraction = {
            "レールウェイ(ポート)": "レールウェイ(フロント)",
            "レールウェイ(フロント)": "レールウェイ(ポート)",
            "スチーマーライン(ハーバー)": "スチーマーライン(ロスト)",
            "スチーマーライン(ロスト)": "スチーマーライン(ハーバー)",
        }

        def swap_attractions(row):
            if row[self.key_from] == row[self.key_to]:
                return move_attraction.get(row[self.key_from], row[self.key_from])
            else:
                return row[self.key_to]

        df_step[self.key_from] = df_step[self.key_from].replace(move_attraction)
        df_step[self.key_to] = df_step.apply(swap_attractions, axis=1)
        return df_step


class StepRegression:
    """座標間距離から歩数を予測する単回帰するクラス"""

    key_pred: str = "steps"

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> RegressionFitResult:
        logger.info("歩数予測モデルの学習")
        self.model = LinearRegression().fit(X=X, y=y)
        return RegressionFitResult(
            coef=self.model.coef_[0][0],
            intercept=self.model.intercept_[0],
            R2=round(self.model.score(X, y), 3),
        )

    def predict(self, X: pd.DataFrame):
        logger.info("座標間距離から歩数予測")
        pred = self.model.predict(X)
        return pred

    def run(self, Input: RegressionInput) -> RegressionResult:
        X, y = Input.get_train()
        result = self.fit(X, y)
        pred_X = Input.get_pred()
        df_pred = Input.df_pred
        df_pred[self.key_pred] = self.predict(pred_X)
        df_train = Input.df_train

        return RegressionResult(df_train=df_train, df_pred=df_pred, fit_result=result)

    def visualize(self, path_folder: str | Path, df_train: pd.DataFrame, df_pred: pd.DataFrame):
        plt.scatter(df_train["calc"], df_train[self.key_pred])
        plt.plot(df_pred["calc"], df_pred[self.key_pred], color="orange")
        plt.xlabel("calc")
        plt.ylabel("actual")
        plt.savefig(path_folder / "regression.png")
