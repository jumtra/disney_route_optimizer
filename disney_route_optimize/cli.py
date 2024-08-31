from pathlib import Path

from omegaconf import OmegaConf

from disney_route_optimize.common.arg_parser import common_parser
from disney_route_optimize.common.config_manager import ConfigManager
from disney_route_optimize.common.count_cpu_core import get_cpu_core
from disney_route_optimize.common.log_handler import add_log_handler
from disney_route_optimize.data.scraping import scraping
from disney_route_optimize.src.route_optimize.optimize import optimize
from disney_route_optimize.src.route_optimize.visualize import opt_visualize
from disney_route_optimize.src.wait_predction.train import Trainer
from disney_route_optimize.src.wait_predction.visualize import visualize
from disney_route_optimize.wait_predction.dataclass.do_cluster import Cluster
from disney_route_optimize.wait_predction.dataclass.do_feature import Features
from disney_route_optimize.wait_predction.dataclass.do_predict import Predictor
from disney_route_optimize.wait_predction.dataclass.do_preprocess import Preprocess


def cli_args(config_manager: ConfigManager) -> ConfigManager:
    """cli argsを反映"""
    parser = common_parser()._get_kwargs()
    for args, kwargs in parser:
        if kwargs is not None:
            config_manager.config.common[args] = kwargs
    return config_manager


def set_cfg(config_manager: ConfigManager) -> ConfigManager:
    """cli argsを反映"""
    # cliを反映
    config_manager = cli_args(config_manager)

    # cpucoreを自動取得
    config_manager.config.common.cpu_core = get_cpu_core(level=config_manager.config.common.cpu_level)

    # 学習済みモデルを使用
    if config_manager.config.common.load_from is not None:
        path_load = Path(str(config_manager.config.common.load_from))
        load_cfg = OmegaConf.load(path_load / "config.yaml")
        config_manager.config.output.wp_output.path_model = load_cfg.output.wp_output.path_model
        config_manager.config.output.wp_output.path_clustering_model = load_cfg.output.wp_output.path_clustering_model
        config_manager.config.output.wp_output.path_clustering_dict = load_cfg.output.wp_output.path_clustering_dict
        config_manager.config.preprocess = load_cfg.preprocess
        config_manager.config.clustering = load_cfg.clustering
        config_manager.config.feature = load_cfg.feature
        config_manager.config.valid = load_cfg.valid
        config_manager.config.predict = load_cfg.predict

        config_manager.config.common.is_only_predict = True
        config_manager.config.common.is_train_predict = False
        config_manager.config.common.is_valid_predict = False
    return config_manager


def cli(config_manager: ConfigManager) -> str:
    config_manager = set_cfg(config_manager)
    train_start_date = config_manager.config.common.train_start_date
    predict_date = config_manager.config.common.predict_date
    folder_name = config_manager.config.common.folder_name
    path_result = Path(config_manager.config.output.output_dir) / f"from{train_start_date}_to{predict_date}_{folder_name}"
    config_manager.config.output.output_dir = str(path_result)
    path_wp_result = Path(config_manager.config.output.wp_output.path_wp_dir)
    path_wp_result.mkdir(exist_ok=True, parents=True)
    path_opt_result = Path(config_manager.config.output.opt_output.path_opt_dir)
    path_opt_result.mkdir(exist_ok=True, parents=True)

    logging = add_log_handler(path_result)
    # データ取得
    scraping(Path(config_manager.config.input.input_dir))

    # 待ち時間予測

    # データ前処理
    pdo = Preprocess(config_manager=config_manager)

    cls_o = Cluster(config_manager=config_manager)

    # cluter情報を加えたwaittimeに更新
    pdo.df_waittime = cls_o.df

    # 特徴量を作成
    feature_obj = Features(config_manager=config_manager, preprocessed_obj=pdo)

    # モデルを学習
    trainer = Trainer(config_manager=config_manager, feature=feature_obj)
    trainer.train()

    logging.info("待ち時間を予測")

    predictor = Predictor(config_manager=config_manager, features=feature_obj)

    if (config_manager.config.common.is_train_predict) and (config_manager.config.common.is_valid_predict):
        logging.info("学習結果を可視化")
        visualize(config_manager=config_manager, predictor=predictor)
    else:
        logging.info("可視化をskip")

    logging.info("待ち時間予測処理を終了")

    # 最適化
    select_land = config_manager.config.common.select_land
    max_num = int(config_manager.config.rank.split_rank)
    if select_land in ["both", "tds"]:
        config_manager.config.rank.split_rank = max_num
        config_manager.config.common.land_type = "tds"
        df_plan, df_pos = optimize(config_manager=config_manager)
        opt_visualize(config_manager=config_manager, df_result=df_plan, df_pos=df_pos)

    # 最適化段階のリセット
    if select_land in ["both", "tdl"]:
        config_manager.config.rank.split_rank = max_num
        config_manager.config.common.land_type = "tdl"
        df_plan, df_pos = optimize(config_manager=config_manager)
        opt_visualize(config_manager=config_manager, df_result=df_plan, df_pos=df_pos)
    config_manager.save_yaml(Path(path_result / "config.yaml"))


if __name__ == "__main__":
    config_dir = "disney_route_optimize/config"
    wp_config_path = "wp_config.yaml"
    opt_config_path = "opt_config.yaml"
    common_config_path = "common_config.yaml"
    config_manager = ConfigManager.from_yaml(
        config_wp_yaml_path=wp_config_path,
        config_opt_yaml_path=opt_config_path,
        common_yaml_path=common_config_path,
        config_dir=config_dir,
    )
    cli(config_manager)
