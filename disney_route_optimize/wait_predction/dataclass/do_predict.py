import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from disney_route_optimize.common.config_manager import ConfigManager
from disney_route_optimize.wait_predction.dataclass.do_feature import Features
from disney_route_optimize.wait_predction.model.lightgbm import LGBM

logger = logging.getLogger(__name__)


@dataclass
class Predictor:
    """予測結果を保持するデータクラス"""

    config_manager: ConfigManager
    features: Features

    def __post_init__(self):
        predict_dir = Path(self.config_manager.config.output.wp_output.path_predict_dir)
        path_train = predict_dir / self.config_manager.config.output.wp_output.pred_train_file
        path_valid = predict_dir / self.config_manager.config.output.wp_output.pred_valid_file
        path_test = predict_dir / self.config_manager.config.output.wp_output.pred_test_file
        if self.config_manager.config.tasks.wp_task.do_predict or (not path_test.exists()):
            self.model = LGBM(self.config_manager)
            self.model.load_model(Path(self.config_manager.config.output.wp_output.path_model))
            cluster_num = self.model.cluster_num
            predict_dir.mkdir(exist_ok=True, parents=True)

            if self.config_manager.config.common.is_train_predict and (not path_train.exists()):
                logger.info("学習データの予測")
                df_train = self._get_predict_features(cluster_num=cluster_num, dict_data=self.features.dict_train)
                df_train.to_csv(path_train, index=False)
                self.df_train = pd.read_csv(path_train)
            else:
                self.df_train = pd.DataFrame()
                logger.info("学習データの予測をskip")

            if self.config_manager.config.common.is_valid_predict and (not path_valid.exists()):
                logger.info("評価データの予測")
                df_valid = self._get_predict_features(cluster_num=cluster_num, dict_data=self.features.dict_valid)
                df_valid.to_csv(path_valid, index=False)
                self.df_valid = pd.read_csv(path_valid)
            else:
                self.df_valid = pd.DataFrame()
                logger.info("評価データの予測をskip")

            logger.info("テストデータの予測")
            df_test = self._get_predict_features(cluster_num=cluster_num, dict_data=self.features.dict_test)
            df_test["date"] = df_test["date"] + timedelta(days=self.config_manager.config.predict.predict_day)
            df_test.to_csv(path_test, index=False)

        else:
            logger.info("予測をskip")
        if self.config_manager.config.common.is_train_predict or path_train.exists():
            self.df_train = pd.read_csv(path_train)
        if self.config_manager.config.common.is_valid_predict or path_valid.exists():
            self.df_valid = pd.read_csv(path_valid)

        return

    def _get_predict_features(self, cluster_num, dict_data):
        def process_cluster(cluster_i):
            if cluster_i not in list(dict_data.keys()):
                logger.info(f"クラスタ番号{cluster_i}のモデルで予測をskip")
                return None

            col_feat = dict_data[cluster_i]["X"].columns
            logger.info(f"クラスタ番号{cluster_i}のモデルで予測")
            df = pd.concat(
                [
                    dict_data[cluster_i]["info"],
                    dict_data[cluster_i]["X"],
                    dict_data[cluster_i]["y"],
                ],
                axis=1,
            )

            target_col = [col for col in df.columns if "recently" in col]
            dict_col2i = {col: i for i, col in enumerate(list(df.columns))}
            target_i = [dict_col2i[col] for col in target_col]
            feat_i = [dict_col2i[col] for col in col_feat]

            list_df = []
            for _, df_g in tqdm(
                df.groupby(["featcat_year", "featcat_month", "featcat_day", "featcat_attraction"]),
                desc=f"クラスター{cluster_i}モデルの予測:",
            ):
                df_temp = df_g.sort_values("featcat_numtime")
                list_pred = []

                for row in df_temp.values:
                    list_feat = self._get_feat(list_pred, recently_num=self.config_manager.config.feature.recently_num)
                    row[target_i] = list_feat
                    list_pred.append(self.model.predict(cluster_i, np.array([row[feat_i]]))[0])
                df_temp["pred"] = list_pred
                list_df.append(df_temp)

            df_result = pd.concat(list_df).reset_index(drop=True)
            return df_result

        list_df_result = Parallel(n_jobs=-1)(delayed(process_cluster)(cluster_i) for cluster_i in range(cluster_num))
        df_result = pd.concat([df for df in list_df_result if df is not None])
        list_col = [col for col in list(df_result.columns) if "feat" not in col] + ["featcat_numtime"]
        df_result = df_result[list_col].rename(columns={"featcat_numtime": "numtime"})

        return df_result

    def _get_feat(self, list_pred: list[float], recently_num: int):
        """直近の予測値を用いる特徴量"""
        list_feat = []
        if len(list_pred) < recently_num:
            missing_num = recently_num - len(list_pred)
            list_recently = [np.nan for _ in range(missing_num)] + list_pred
        else:
            list_recently = list_pred[len(list_pred) - recently_num :]

        list_feat.extend(list_recently)
        list_feat.append(np.nanmean(list_recently))
        list_feat.append(np.nanmin(list_recently))
        list_feat.append(np.nanmax(list_recently))
        list_feat.append(np.nanstd(list_recently))

        return list_feat
