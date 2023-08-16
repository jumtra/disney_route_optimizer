import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from disney_route_optimize.common.config_maneger import ConfigManeger
from disney_route_optimize.wait_predction.dataclass.do_preprocess import Preprocess
from disney_route_optimize.wait_predction.preprocess.features.make_features import make_features

logger = logging.getLogger(__name__)


@dataclass
class Features:
    config_maneger: ConfigManeger
    preprocessed_obj: Preprocess
    key_target: str = "target"

    def __post_init__(self):
        path_feat = Path(self.config_maneger.config.output.wp_output.features_file)
        path_test = Path(self.config_maneger.config.output.wp_output.features_test_file)

        predict_start_date = datetime.strptime(
            self.config_maneger.config.common.predict_date, "%Y-%m-%d"
        )
        if (
            self.config_maneger.config.tasks.wp_task.do_make_feat
            or not path_feat.exists()
            or not path_test.exists()
        ):
            logger.info("特徴量を作成")
            Path(self.config_maneger.config.output.wp_output.path_features_dir).mkdir(
                exist_ok=True, parents=True
            )
            df, df_test = make_features(
                df_waittime=self.preprocessed_obj.df_waittime,
                df_weather=self.preprocessed_obj.df_weather,
                predict_start_date=predict_start_date,
                config_manager=self.config_maneger,
            )
            df.to_csv(path_feat, index=False)
            df_test.to_csv(path_test, index=False)
        else:
            logger.info("特徴量作成をskip")
        # dataframe 読み込み
        df = pd.read_csv(path_feat)
        df_test = pd.read_csv(path_test)
        # type annotate
        df["date"] = pd.to_datetime(df["date"])
        dict_dtype = {feat: "category" for feat in list(df.columns) if "featcat_" in feat}
        dict_dtype.update({feat: "float32" for feat in list(df.columns) if "featnum_" in feat})
        dict_dtype.update({feat: "bool" for feat in list(df.columns) if "featbool_" in feat})
        self.df_feat = df.astype(dict_dtype)

        df_test["date"] = pd.to_datetime(df_test["date"])
        dict_dtype = {feat: "category" for feat in list(df_test.columns) if "featcat_" in feat}
        dict_dtype.update(
            {feat: "float32" for feat in list(df_test.columns) if "featnum_" in feat}
        )
        dict_dtype.update({feat: "bool" for feat in list(df_test.columns) if "featbool_" in feat})
        self.df_test = df_test.astype(dict_dtype)

        list_feat = [feat for feat in list(df.columns) if "feat" in feat]
        valid_date = df["date"].max() - timedelta(
            days=self.config_maneger.config.features.valid_days
        )
        df_train = df.loc[df["date"] < valid_date]
        df_valid = df.loc[df["date"] >= valid_date]

        self.dict_train = self._to_dict(df=df_train, list_feat=list_feat)
        self.dict_valid = self._to_dict(df=df_valid, list_feat=list_feat)
        self.dict_test = self._to_dict(df=df_test, list_feat=list_feat)

    def _to_dict(self, df: pd.DataFrame, list_feat: list[str]) -> dict:
        dict_cluster2df = defaultdict(lambda: np.nan)
        for cluster, df in df.groupby("cluster"):
            dict_cluster2df[cluster] = {
                "info": df.drop(list_feat + [self.key_target], axis=1),
                "X": df[list_feat],
                "y": df[self.key_target],
            }

        return dict_cluster2df
