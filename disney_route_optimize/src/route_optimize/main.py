from pathlib import Path

from disney_route_optimize.common.config_maneger import ConfigManeger
from disney_route_optimize.common.log_handler import add_log_handler
from disney_route_optimize.src.route_optimize.optimize import optimize


def main():
    config_dir = "disney_route_optimize/config"
    wp_config_path = "wp_config.yaml"
    opt_config_path = "opt_config.yaml"
    common_config_path = "common_config.yaml"
    config_maneger = ConfigManeger.from_yaml(
        config_wp_yaml_path=wp_config_path,
        config_opt_yaml_path=opt_config_path,
        common_yaml_path=common_config_path,
        config_dir=config_dir,
    )

    config_maneger.config.common.land_type = "tdl"
    config_maneger.config.output.output_dir = "result/from2022-04-25_to2023-07-26_both_v1"
    path_result = Path(config_maneger.config.output.opt_output.path_opt_dir)
    path_result.mkdir(exist_ok=True, parents=True)
    logging = add_log_handler(path_result)
    optimize(config_maneger=config_maneger)


if __name__ == "__main__":
    main()
