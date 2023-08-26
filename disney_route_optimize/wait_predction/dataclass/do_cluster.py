import logging
import pickle
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd

from disney_route_optimize.common.config_manager import ConfigManager
from disney_route_optimize.wait_predction.model.clustering import Clustering

logger = logging.getLogger(__name__)


@dataclass
class Cluster:
    """kshapeのクラスタリング結果を保存するデータクラス"""

    config_manager: ConfigManager

    def __post_init__(self):
        clustering_dir = Path(self.config_manager.config.output.wp_output.path_clustering_dir)
        path_model = Path(self.config_manager.config.output.wp_output.path_clustering_model)
        path_clustered_csv = Path(self.config_manager.config.output.wp_output.path_clustered_waittime)
        path_cluster_dict = Path(self.config_manager.config.output.wp_output.path_clustering_dict)
        self.model = Clustering(self.config_manager)
        path_waittime = Path(self.config_manager.config.output.wp_output.waittime_file)
        is_only_predict = self.config_manager.config.common.is_only_predict
        if self.config_manager.config.tasks.wp_task.do_clustering or (not path_model.exists()) or is_only_predict:
            logger.info("時系列クラスタリング")
            clustering_dir.mkdir(exist_ok=True, parents=True)
            path_csv = self.config_manager.config.output.wp_output.waittime_file
            df = pd.read_csv(path_csv)
            df["date"] = pd.to_datetime(df["date"])
            valid_date = df["date"].max() - timedelta(days=self.config_manager.config.valid.valid_days)
            df_train = df.loc[df["date"] < valid_date]

            if is_only_predict:
                logger.info("学習をskip")
                self.model.load_model()

                with open(path_cluster_dict, "rb") as file:
                    dict_attr2cluster = pickle.load(file)
            else:
                dict_attr2cluster = self.model.train(df_train)
                self.model.save_model()

            df = self.model.add_cluster_col(df, dict_attr2cluster)
            df.to_csv(path_clustered_csv)

            with open(path_cluster_dict, "wb") as file:
                pickle.dump(dict_attr2cluster, file)

        else:
            logger.info("時系列クラスタリングをskip")

        self.model.load_model()

        with open(path_cluster_dict, "rb") as file:
            self.dict_attr2cluster = pickle.load(file)
        df_waittime = pd.read_csv(path_waittime)
        self.df = self.model.add_cluster_col(df_waittime, self.dict_attr2cluster)

        return
