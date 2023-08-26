from pathlib import Path

from disney_route_optimize.common.config_manager import ConfigManager
from disney_route_optimize.route_optimize.dataclass.do_cost import CostMatrix
from disney_route_optimize.route_optimize.dataclass.do_distance import AttractionStepDistance
from disney_route_optimize.route_optimize.dataclass.do_position import AttractionPosition
from disney_route_optimize.route_optimize.dataclass.do_predict_result import PredictResult
from disney_route_optimize.route_optimize.dataclass.do_rank import AttractionRank
from disney_route_optimize.route_optimize.dataclass.do_time import AttractionTime
from disney_route_optimize.route_optimize.preprocess.step_regression import RegressionInput, StepRegression


def calc_cost(config_manager: ConfigManager) -> tuple[CostMatrix, CostMatrix]:
    """優先最適化対象のコスト行列と次点最適化対象のコスト行列を取得する関数"""
    pred = PredictResult(config_manager=config_manager)
    attr_pos = AttractionPosition(config_manager=config_manager)
    attr_time = AttractionTime(config_manager=config_manager)
    target_attraction = attr_pos.target_attraction
    attr_dis = AttractionStepDistance(config_manager=config_manager, target_attraction=target_attraction)

    path_regression = Path(config_manager.config.output.opt_output.path_regression_dir)
    path_regression.mkdir(parents=True, exist_ok=True)
    input = RegressionInput(attraction_distance=attr_dis, attraction_position=attr_pos)

    result = StepRegression().run(input)
    StepRegression().visualize(path_folder=path_regression, df_train=result.df_train, df_pred=result.df_pred)
    result.fit_result.to_df().to_csv(path_regression / config_manager.config.output.opt_output.path_regression_file, index=False)

    attr_rank = AttractionRank(config_manager=config_manager)
    cost_first = CostMatrix(
        list_target_cols=attr_rank.list_first_rank,
        dict_attraction2rank=attr_rank.dict_attraction2rank,
        config_manager=config_manager,
        df_pred=pred.df_pred,
        df_step=result.df_steps,
        dict_attraction2time=attr_time.dict_attraction2time,
    )
    cost_second = CostMatrix(
        list_target_cols=attr_rank.list_second_rank,
        dict_attraction2rank=attr_rank.dict_attraction2rank,
        config_manager=config_manager,
        df_pred=pred.df_pred,
        df_step=result.df_steps,
        dict_attraction2time=attr_time.dict_attraction2time,
    )
    return cost_first, cost_second
