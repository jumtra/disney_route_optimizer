from pathlib import Path

from disney_route_optimize.common.config_maneger import ConfigManeger
from disney_route_optimize.common.log_handler import add_log_handler
from disney_route_optimize.src.wait_predction.train import Trainer
from disney_route_optimize.src.wait_predction.visualize import visualize
from disney_route_optimize.wait_predction.dataclass.do_cluster import Cluster
from disney_route_optimize.wait_predction.dataclass.do_feature import Features
from disney_route_optimize.wait_predction.dataclass.do_predict import Predictor
from disney_route_optimize.wait_predction.dataclass.do_preprocess import Preprocess

config_dir = "disney_route_optimize/config"
wp_config_path = "wp_config.yaml"
common_config_path = "common_config.yaml"


def main():
    config_maneger = ConfigManeger.from_yaml(
        config_yaml_path=wp_config_path,
        common_yaml_path=common_config_path,
        config_dir=config_dir,
    )

    if config_maneger.config.wp_model.model_path is not None:
        path_result = Path(str(config_maneger.config.wp_model.model_path))
    else:
        train_start_date = config_maneger.config.common.train_start_date
        predict_date = config_maneger.config.common.predict_date
        folder_name = config_maneger.config.common.folder_name
        path_result = (
            Path(config_maneger.config.output.output_dir)
            / f"from{train_start_date}_to{predict_date}_{folder_name}"
        )
    config_maneger.config.output.output_dir = str(path_result)
    path_result = Path(config_maneger.config.output.wp_output.path_wp_dir)
    path_result.mkdir(exist_ok=True, parents=True)

    if config_maneger.config.wp_model.model_path is None:
        logging = add_log_handler(path_result)
    else:
        logging = add_log_handler("./")
    # データ前処理
    pdo = Preprocess(config_maneger=config_maneger)

    cls_o = Cluster(config_maneger=config_maneger)

    # cluter情報を加えたwaittimeに更新
    pdo.df_waittime = cls_o.df

    # 特徴量を作成
    feature_obj = Features(config_maneger=config_maneger, preprocessed_obj=pdo)

    # モデルを学習
    trainer = Trainer(config_maneger=config_maneger, feature=feature_obj)
    trainer.train()

    logging.info("待ち時間を予測")

    predictor = Predictor(config_maneger=config_maneger, features=feature_obj)

    logging.info("学習結果を可視化")
    visualize(config_maneger=config_maneger, predictor=predictor)
    logging.info("結果を保存")

    if config_maneger.config.wp_model.model_path is None:
        config_maneger.save_yaml(Path(path_result / "config.yaml"))


if __name__ == "__main__":
    main()