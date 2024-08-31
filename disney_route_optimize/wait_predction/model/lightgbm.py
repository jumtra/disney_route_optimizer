import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd

from disney_route_optimize.common.config_manager import ConfigManager

from .metrics import rmsle
from .save_model import get_bestiter_model

logger = logging.getLogger(__name__)


@dataclass
class ClusteredModelCollection:
    list_model: list[lgb.Booster] = field(default_factory=list)
    list_eval_result: list[dict] = field(default_factory=list)

    def add_cluster_train_result(self, model: lgb.Booster, eval_result: dict) -> None:
        self.list_model.append(model)
        self.list_eval_result.append(eval_result)


class LGBM:
    def __init__(self, config_manager: ConfigManager):
        self.model = None
        self.config_manager = config_manager
        self._clustered_model_collection: Optional(ClusteredModelCollection) = None
        self._set_cfg()

    def _set_cfg(self):
        self.params = dict(self.config_manager.config.wp_model.regression_params.params)
        self.early_stopping_rounds = self.config_manager.config.wp_model.regression_params.early_stopping_rounds  # アーリーストッピング設定
        self.log_eval = self.config_manager.config.wp_model.regression_params.log_eval
        self.cluster_num = self.config_manager.config.clustering.n_clusters

    def _get_metrics(self):
        metrics_name = self.config_manager.config.wp_model.regression_params.custom_metrics
        dict_metrics = {"rmsle": rmsle}
        metrics = dict_metrics.get(metrics_name, None)
        return metrics

    def train_model(self, X_train, y_train, X_valid, y_valid) -> tuple[lgb.Booster, dict]:
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
        history = {}
        metrics = self._get_metrics()

        model = lgb.train(
            params=self.params,  # ハイパーパラメータをセット
            train_set=lgb_train,  # 訓練データを訓練用にセット
            valid_sets=[lgb_train, lgb_valid],  # 訓練データとテストデータをセット
            valid_names=["Train", "Valid"],  # データセットの名前をそれぞれ設定
            feval=metrics,
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=True),
                lgb.log_evaluation(self.log_eval),
                lgb.record_evaluation(history),
            ],
        )
        model = get_bestiter_model(model=model)
        return model, history

    def train(self, dict_train, dict_valid) -> None:
        self._clustered_model_collection = ClusteredModelCollection()

        for i_cluster in range(self.cluster_num):
            dict_valid_Xy = dict_valid[i_cluster]
            dict_train_Xy = dict_train[i_cluster]
            train_X = dict_train_Xy["X"]
            train_y = dict_train_Xy["y"]
            valid_X = dict_valid_Xy["X"]
            valid_y = dict_valid_Xy["y"]

            logger.info(f"クラスタ番号{i_cluster}のモデルを学習")

            model, eval_result = self.train_model(train_X, train_y, valid_X, valid_y)

            self._clustered_model_collection.add_cluster_train_result(model=model, eval_result=eval_result)
        logger.info("学習処理終了")

        return

    def save_model(self, path_model: Path):
        dump_items = (
            self._clustered_model_collection.list_model,
            self._clustered_model_collection.list_eval_result,
            self.cluster_num,
        )
        with open(path_model, "wb") as file:
            pickle.dump(dump_items, file)

        for cluster_num, (model, eval_result) in enumerate(
            zip(
                self._clustered_model_collection.list_model,
                self._clustered_model_collection.list_eval_result,
            )
        ):
            try:
                self.save_for_analytics(model=model, history=eval_result, cluster_num=cluster_num)
            except Exception:
                logger.warning(f"model cluster {cluster_num}の可視化に失敗しました。")

    def load_model(self, model_path: Path):
        self._clustered_model_collection = ClusteredModelCollection()
        with open(model_path, "rb") as file:
            (
                self._clustered_model_collection.list_model,
                self._clustered_model_collection.list_eval_result,
                self.cluster_num,
            ) = pickle.load(file)

    def save_for_analytics(self, model, history, cluster_num):
        train_dir = Path(self.config_manager.config.output.wp_output.path_train_dir)
        # 特徴量重要度
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)
        lgb.plot_importance(model, max_num_features=20, ax=ax, title="Feature_importance")
        plt.yticks(rotation=45)
        fig.savefig(train_dir / f"feature_importance_clsuter{cluster_num}.png")
        plt.cla()
        plt.clf()
        plt.close("all")

        # csvで保存
        # 特徴量の重要度を取得
        feature_importance = model.feature_importance()
        # 特徴量名を取得
        feature_names = model.feature_name()
        # 特徴量の重要度をDataFrameに変換
        df_importance = pd.DataFrame({"feature": feature_names, "importance": feature_importance})

        df_importance.to_csv(train_dir / f"feature_importance_cluster{cluster_num}.csv", index=False)

        # metric
        fig, ax = plt.subplots()
        lgb.plot_metric(history, ax=ax, title="train-valid-metric")
        fig.savefig(train_dir / f"train-valid-metric_cluster{cluster_num}.png")
        plt.cla()
        plt.clf()
        plt.close("all")

    def predict(self, cluster_num: int, test_X):
        pred = self._clustered_model_collection.list_model[cluster_num].predict(test_X)
        return pred
