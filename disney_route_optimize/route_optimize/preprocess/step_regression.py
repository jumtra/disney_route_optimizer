from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ..dataclass.do_distance import AttractionStepDistance
from ..dataclass.do_position import AttractionPosition
from ..preprocess.calc_distance import calc_distance


@dataclass
class RegressionInput:
    attraction_position: AttractionPosition
    attraction_distance: AttractionStepDistance
    key_actual: str = "steps"  # 実測値
    key_calc_distance: str = "calc"  # 座標間距離

    def __post_init__(self):
        self.df_train = self.make_train_input()
        self.df_pred = self.make_predict_input()

    def make_train_input(self) -> pd.DataFrame:
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
        X = self.df_train[[self.key_calc_distance]]
        y = self.df_train[[self.key_actual]]
        return X, y

    def get_pred(self):
        X = self.df_pred[[self.key_calc_distance]]
        return X


@dataclass
class RegressionFitResult:
    coef: float
    intercept: float
    R2: float


@dataclass
class RegressionResult:
    df_train: pd.DataFrame
    df_pred: pd.DataFrame
    fit_result: RegressionFitResult

    def __post_init__(self):
        self.df_steps = pd.concat([self.df_pred, self.df_train])


class StepRegression:
    """座標間距離から歩数を予測する単回帰するクラス"""

    key_pred: str = "steps"

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> RegressionFitResult:
        self.model = LinearRegression().fit(X=X, y=y)
        return RegressionFitResult(
            coef=self.model.coef_,
            intercept=self.model.intercept_,
            R2=round(self.model.score(X, y), 3),
        )

    def predict(self, X: pd.DataFrame):
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

    def visualize(self, df_reg: pd.DataFrame):
        plt.scatter(df_reg["calc"], df_reg["actual"])
        plt.plot(df_reg["calc"], df_reg["pred"], color="orange")
        plt.xlabel("calc")
        plt.ylabel("actual")
        plt.show()
