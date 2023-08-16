from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from disney_route_optimize.common.config_maneger import ConfigManeger


def make_feat_clsuter(df: pd.DataFrame, config_maneger: ConfigManeger):
    n_clusters = config_maneger.config.feature.clustering.n_clusters
    n_init = config_maneger.config.feature.clustering.n_clusters
    max_iter = config_maneger.config.feature.clustering.max_iter
    path_clustering = Path(config_maneger.config.output.wp_output.path_clustering_dir)

    dict_attraction_info = defaultdict(lambda: np.nan)
    for attr, df in df.groupby("attraction"):
        temp_list = list(df[["wait_time"]].describe().values.reshape(-1))[1:]
        if not all(np.isnan(temp_list)):
            dict_attraction_info[attr] = temp_list

    target_array = np.array(list(dict_attraction_info.values()))

    sk_km = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter).fit(target_array)
    labels = sk_km.labels_

    dict_attraction2cluster = {
        attr: label for attr, label in zip(list(dict_attraction_info.keys()), labels)
    }

    pca = PCA(n_components=2)  # 2次元に削減
    X_reduced = pca.fit_transform(target_array)

    # 可視化
    fig = plt.figure(figsize=(8, 6))

    for i in range(n_clusters):
        plt.scatter(
            X_reduced[labels == i, 0],
            X_reduced[labels == i, 1],
            label=f"Cluster {i+1}",
        )

    plt.legend()
    plt.savefig(path_clustering / "feature_clustered.png")
    plt.close()

    return dict_attraction2cluster
