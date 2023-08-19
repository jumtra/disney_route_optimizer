from pathlib import Path

from omegaconf import OmegaConf

from disney_route_optimize.common.arg_parser import common_parser
from disney_route_optimize.common.config_maneger import ConfigManeger
from disney_route_optimize.common.count_cpu_core import get_cpu_core
from disney_route_optimize.common.log_handler import add_log_handler
from disney_route_optimize.data.scraping import scraping
from disney_route_optimize.src.route_optimize.optimize import optimize
from disney_route_optimize.src.wait_predction.train import Trainer
from disney_route_optimize.src.wait_predction.visualize import visualize
from disney_route_optimize.wait_predction.dataclass.do_cluster import Cluster
from disney_route_optimize.wait_predction.dataclass.do_feature import Features
from disney_route_optimize.wait_predction.dataclass.do_predict import Predictor
from disney_route_optimize.wait_predction.dataclass.do_preprocess import Preprocess

config_dir = "disney_route_optimize/config"
wp_config_path = "wp_config.yaml"
opt_config_path = "opt_config.yaml"
common_config_path = "common_config.yaml"


def cli_args(config_maneger: ConfigManeger) -> ConfigManeger:
    """cli argsを反映"""
    parser = common_parser()._get_kwargs()
    for args, kwargs in parser:
        if kwargs is not None:
            config_maneger.config.common[args] = kwargs
    return config_maneger


def set_cfg(config_maneger: ConfigManeger) -> ConfigManeger:
    """cli argsを反映"""
    # cliを反映
    config_maneger = cli_args(config_maneger)

    # cpucoreを自動取得
    config_maneger.config.common.cpu_core = get_cpu_core(level=config_maneger.config.common.cpu_level)

    # 学習済みモデルを使用
    if config_maneger.config.common.load_from is not None:
        path_load = Path(str(config_maneger.config.common.load_from))
        load_cfg = OmegaConf.load(path_load / "config.yaml")
        config_maneger.config.output.wp_output.path_model = load_cfg.output.wp_output.path_model
        config_maneger.config.output.wp_output.path_clustering_model = load_cfg.output.wp_output.path_clustering_model
        config_maneger.config.output.wp_output.path_clustering_dict = load_cfg.output.wp_output.path_clustering_dict
        config_maneger.config.preprocess = load_cfg.preprocess
        config_maneger.config.clustering = load_cfg.clustering
        config_maneger.config.feature = load_cfg.feature
        config_maneger.config.valid = load_cfg.valid
        config_maneger.config.predict = load_cfg.predict

        config_maneger.config.common.is_only_predict = True
        config_maneger.config.common.is_train_predict = False
        config_maneger.config.common.is_valid_predict = False
    return config_maneger


def main():
    config_maneger = ConfigManeger.from_yaml(
        config_wp_yaml_path=wp_config_path,
        config_opt_yaml_path=opt_config_path,
        common_yaml_path=common_config_path,
        config_dir=config_dir,
    )
    config_maneger = set_cfg(config_maneger)
    train_start_date = config_maneger.config.common.train_start_date
    predict_date = config_maneger.config.common.predict_date
    folder_name = config_maneger.config.common.folder_name
    path_result = Path(config_maneger.config.output.output_dir) / f"from{train_start_date}_to{predict_date}_{folder_name}"
    config_maneger.config.output.output_dir = str(path_result)
    path_wp_result = Path(config_maneger.config.output.wp_output.path_wp_dir)
    path_wp_result.mkdir(exist_ok=True, parents=True)
    path_opt_result = Path(config_maneger.config.output.opt_output.path_opt_dir)
    path_opt_result.mkdir(exist_ok=True, parents=True)

    logging = add_log_handler(path_result)
    # データ取得
    scraping(Path(config_maneger.config.input.input_dir))

    # 待ち時間予測

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

    if (config_maneger.config.common.is_train_predict) and (config_maneger.config.common.is_valid_predict):
        logging.info("学習結果を可視化")
        visualize(config_maneger=config_maneger, predictor=predictor)
    else:
        logging.info("可視化をskip")

    logging.info("待ち時間予測処理を終了")

    # 最適化
    max_num = int(config_maneger.config.rank.split_rank)
    config_maneger.config.common.land_type = "tds"
    optimize(config_maneger=config_maneger)
    # 最適化段階のリセット
    config_maneger.config.rank.split_rank = max_num
    config_maneger.config.common.land_type = "tdl"
    optimize(config_maneger=config_maneger)

    config_maneger.save_yaml(Path(path_result / "config.yaml"))


if __name__ == "__main__":
    main()
