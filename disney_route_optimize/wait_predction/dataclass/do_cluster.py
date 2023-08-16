import logging
import pickle
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd

from disney_route_optimize.common.config_maneger import ConfigManeger
from disney_route_optimize.wait_predction.model.clustering import Clustering

logger = logging.getLogger(__name__)


@dataclass
class Cluster:
    config_maneger: ConfigManeger

    def __post_init__(self):
        clustering_dir = Path(self.config_maneger.config.output.wp_output.path_clustering_dir)
        path_model = Path(self.config_maneger.config.output.wp_output.path_clustering_model)
        path_clustered_csv = Path(
            self.config_maneger.config.output.wp_output.path_clustered_waittime
        )
        path_cluster_dict = Path(self.config_maneger.config.output.wp_output.path_clustering_dict)
        self.model = Clustering(self.config_maneger)
        path_waittime = Path(self.config_maneger.config.output.wp_output.waittime_file)
        if self.config_maneger.config.tasks.wp_task.do_clustering or (not path_model.exists()):
            logger.info("時系列クラスタリング")
            clustering_dir.mkdir(exist_ok=True, parents=True)
            path_csv = self.config_maneger.config.output.wp_output.waittime_file
            df = pd.read_csv(path_csv)
            df["date"] = pd.to_datetime(df["date"])
            valid_date = df["date"].max() - timedelta(
                days=self.config_maneger.config.features.valid_days
            )
            df_train = df.loc[df["date"] < valid_date]
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
