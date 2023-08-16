from pathlib import Path

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from tslearn.clustering import KShape


class Clustering:
    def __init__(self, config_maneger):
        self.config_maneger = config_maneger
        self.max_len = 48
        self._set_params()
        self.ks = KShape(
            n_clusters=self.n_clusters,
            verbose=self.verbose,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

    def _set_params(self):
        self.n_clusters = self.config_maneger.config.clustering.n_clusters
        self.verbose = self.config_maneger.config.clustering.kshape.verbose
        self.n_init = self.config_maneger.config.clustering.kshape.n_init
        self.max_iter = self.config_maneger.config.clustering.kshape.max_iter
        self.random_state = self.config_maneger.config.common.seed

    def save_model(self) -> None:
        self.ks.to_json(self.config_maneger.config.output.wp_output.path_clustering_model)

    def load_model(self) -> None:
        self.ks = self.ks.from_json(
            self.config_maneger.config.output.wp_output.path_clustering_model
        )

    def core_train(self, X: np.ndarray):
        self.ks.fit(X)
        # クラスタリング結果を取得
        cluster_centers = self.ks.cluster_centers_
        cluster_centers = cluster_centers.reshape(cluster_centers.shape[0], -1)
        labels_train = self.ks.labels_
        return cluster_centers, labels_train

    def _convert_data(
        self,
        df: pd.DataFrame,
        key_date: str = "date",
        key_attraction: str = "attraction",
        key_wait_time: str = "wait_time",
        key_weekday: str = "weekday",
    ) -> tuple[np.ndarray, list[pd.DataFrame]]:
        target_array = np.empty((0, self.max_len))
        list_attr = []
        scaler = MinMaxScaler()
        for attraction_weekday, group_df in df.groupby([key_attraction, key_weekday]):
            attraction_array = np.empty((0, self.max_len))
            for col_date, date_df in group_df.groupby(key_date):
                val = date_df[key_wait_time].values
                if all(np.isnan(val)):
                    continue
                if val.shape[0] < self.max_len:
                    missing_num = self.max_len - val.shape[0]
                    val = np.append(val, np.full(missing_num, np.nan))
                val = np.where(np.isnan(val), -1, val)
                attraction_array = np.vstack((attraction_array, val))
            if not np.all(np.isnan(attraction_array)):
                attraction_array = np.mean(attraction_array, axis=0)
                attraction_array = scaler.fit_transform(attraction_array.reshape(-1, 1)).reshape(
                    1, -1
                )
                list_attr.append(attraction_weekday)
                target_array = np.vstack((target_array, attraction_array))
        return target_array, list_attr

    def train(self, df_train: pd.DataFrame):
        target_array, list_attr = self._convert_data(df=df_train)

        cluster_centers, _ = self.core_train(target_array)

        labels_pred = self.ks.predict(target_array)

        dict_attr2cluster = {
            attraction: cluster_num for attraction, cluster_num in zip(list_attr, labels_pred)
        }
        self.visualize(cluster_centers, labels_pred, target_array, list_attr)
        return dict_attr2cluster

    def add_cluster_col(self, df: pd.DataFrame, dict_attr2cluster):
        # dfにクラスタ番号を割り当てる
        list_all = list(dict_attr2cluster.keys())
        list_attr = list(set([attr[0] for attr in list_all]))
        labels_pred = list(dict_attr2cluster.values())
        df = df.loc[df["attraction"].isin(list_attr)].copy()
        df["cluster"] = np.nan
        for info, cls_num in zip(list_all, labels_pred):
            attraction_name = info[0]
            weekday = info[1]
            df.loc[
                df[(df["attraction"] == attraction_name) & (df["weekday"] == weekday)].index,
                "cluster",
            ] = cls_num
        return df

    def visualize(self, cluster_centers, labels_pred, target_array, list_attr):
        # クラスタセンターをプロット
        plt.figure(figsize=(8, 6))
        for i in range(self.n_clusters):
            plt.plot(cluster_centers[i].ravel(), label=f"Cluster {i+1}")

        plt.title("kshape クラスタリング結果")
        plt.xlabel("時間軸")
        plt.ylabel("値")
        plt.legend()

        plt.savefig(
            Path(self.config_maneger.config.output.wp_output.path_clustering_dir)
            / "kshape_centroid.png"
        )
        plt.close()

        # クラスタ毎の時系列をプロット
        for c in range(self.n_clusters):
            plt.figure(figsize=(8, 6))
            for i in range(len(labels_pred)):
                if labels_pred[i] == c:
                    plt.plot(target_array[i].ravel(), label=f"{list_attr[i]}")

            plt.title(f"kshape cluster_{c}")
            plt.xlabel("時間軸")
            plt.ylabel("値")
            # plt.legend()

            plt.savefig(
                Path(self.config_maneger.config.output.wp_output.path_clustering_dir)
                / f"cluster_{c}.png"
            )
            plt.close()
