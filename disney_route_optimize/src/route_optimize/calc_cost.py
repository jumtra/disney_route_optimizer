from disney_route_optimize.common.config_maneger import ConfigManeger
from disney_route_optimize.route_optimize.dataclass.do_cost import CostMatrix
from disney_route_optimize.route_optimize.dataclass.do_distance import AttractionStepDistance
from disney_route_optimize.route_optimize.dataclass.do_position import AttractionPosition
from disney_route_optimize.route_optimize.dataclass.do_predict_result import PredictResult
from disney_route_optimize.route_optimize.dataclass.do_rank import AttractionRank
from disney_route_optimize.route_optimize.dataclass.do_time import AttractionTime
from disney_route_optimize.route_optimize.preprocess.step_regression import RegressionInput, StepRegression


def calc_cost(config_maneger: ConfigManeger) -> tuple[CostMatrix, CostMatrix]:
    pred = PredictResult(config_maneger=config_maneger)
    attr_pos = AttractionPosition(config_maneger=config_maneger)
    attr_time = AttractionTime(config_maneger=config_maneger)
    attr_rank = AttractionRank(config_maneger=config_maneger)
    target_attraction = attr_pos.target_attraction
    attr_dis = AttractionStepDistance(config_maneger=config_maneger, target_attraction=target_attraction)

    input = RegressionInput(attraction_distance=attr_dis, attraction_position=attr_pos)

    result = StepRegression().run(input)

    cost_first = CostMatrix(
        list_target_cols=attr_rank.list_first_rank,
        config_maneger=config_maneger,
        df_pred=pred.df_pred,
        df_step=result.df_steps,
        dict_attraction2time=attr_time.dict_attraction2time,
    )
    cost_second = CostMatrix(
        list_target_cols=attr_rank.list_second_rank,
        config_maneger=config_maneger,
        df_pred=pred.df_pred,
        df_step=result.df_steps,
        dict_attraction2time=attr_time.dict_attraction2time,
    )
    return cost_first, cost_second
