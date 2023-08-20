import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

from disney_route_optimize.common.config_maneger import ConfigManeger
from disney_route_optimize.route_optimize.dataclass.do_cost import CostMatrix
from disney_route_optimize.route_optimize.model.get_constraint import get_constraint
from disney_route_optimize.route_optimize.model.get_objective import get_objective
from disney_route_optimize.route_optimize.model.variable import Variable
from disney_route_optimize.src.route_optimize.calc_cost import calc_cost

logger = logging.getLogger(__name__)


def optimize(config_maneger: ConfigManeger) -> None:
    path_optimize = config_maneger.config.output.opt_output.path_opt_dir
    if config_maneger.config.tasks.opt_task.do_optimize or not Path(path_optimize).exists():
        logger.info("最適化を実行")
        cost_first, cost_second = calc_cost(config_maneger=config_maneger)

        # Step1 優先最適化
        is_not_solve = True

        # 優先アトラクションの数を減らしながら最適化
        while is_not_solve:
            logger.info(f"Step1 優先最適化対象：{config_maneger.config.rank.split_rank} で実行")
            is_not_solve = optimize_core(config_maneger=config_maneger, cost=cost_first, is_first=True)
            if is_not_solve:
                config_maneger.config.rank.split_rank -= 1
                cost_first, cost_second = calc_cost(config_maneger=config_maneger)
        # Step2 次点最適化
        # TODO 他に追加できるアトラクションを探索
    else:
        logger.info("最適化をskip")


def optimize_core(config_maneger: ConfigManeger, cost: CostMatrix, is_first: bool) -> bool:
    """最適化のコア部分"""
    is_not_solve = True
    list_cost = cost.list_cost
    list_locations = cost.list_target_cols
    model = pulp.LpProblem("ParkTSP", pulp.LpMaximize)
    opt_var = Variable(cost_list=list_cost, len_times=len(list_cost), len_locations=len(list_locations))

    # constraint params
    return_time = pd.to_datetime(config_maneger.config.common.return_time)
    visit_time = pd.to_datetime(config_maneger.config.common.visit_time)
    global_max_time = (return_time - visit_time) / timedelta(days=1) * 24 * 60
    sep_time = config_maneger.config.cost.sep_time

    # TODO: 綺麗に書き直す
    if is_first:
        path_optimize = config_maneger.config.output.opt_output.path_first_optimize_dir
        # 制約の追加
        for const_name, is_use in config_maneger.config.first_opt.constraints.items():
            if is_use:
                model = get_constraint(constraint_name=const_name)(
                    model=model, variable=opt_var, sep_time=sep_time, global_time=global_max_time
                ).get_model()
        # 目的関数の追加
        for objective_name, is_use in config_maneger.config.first_opt.objective.items():
            if is_use:
                model = get_objective(objective_name=objective_name)(model=model, variable=opt_var, list_rank=cost.list_rank).get_model()

    else:
        path_optimize = config_maneger.config.output.opt_output.path_second_optimize_dir
        # 制約の追加
        for const_name, is_use in config_maneger.config.first_opt.constraints.items():
            if is_use:
                model = get_constraint(constraint_name=const_name)(
                    model=model, variable=opt_var, sep_time=sep_time, global_time=global_max_time
                ).get_model()
        # 目的関数の追加
        for objective_name, is_use in config_maneger.config.first_opt.objective.items():
            if is_use:
                model = get_objective(objective_name=objective_name)(model=model, variable=opt_var, list_rank=cost.list_rank).get_model()

    model.solve(pulp.PULP_CBC_CMD(**config_maneger.config.opt_params))

    if pulp.LpStatus[model.status] not in ["Not Solved", "Infeasible"]:
        # 結果を表示
        logger.info(f"最適化結果：{pulp.LpStatus[model.status]}")

        df_time = _get_time_matrix(list_cost=list_cost, list_locations=list_locations, opt_var=opt_var)

        df_location = _get_location_matrix(list_cost=list_cost, list_locations=list_locations, opt_var=opt_var)
        df_plan = _get_plan(
            cost=cost, df_time=df_time, list_locations=list_locations, opt_var=opt_var, visit_time=visit_time, return_time=return_time
        )

        logger.info("最適化結果を保存")
        path_optimize = Path(path_optimize) / str(config_maneger.config.common.land_type)
        Path(path_optimize).mkdir(parents=True, exist_ok=True)
        df_time.to_csv(Path(path_optimize) / config_maneger.config.output.opt_output.path_time_file, index=False)
        df_location.to_csv(Path(path_optimize) / config_maneger.config.output.opt_output.path_location_file, index=False)
        df_plan.to_csv(Path(path_optimize) / config_maneger.config.output.opt_output.path_plan_file, index=False)
        is_not_solve = False
    else:
        logger.info("最適化結果：解が見つかりませんでした")
    return is_not_solve


def _get_location_matrix(list_cost: list[np.ndarray], list_locations: list[str], opt_var: Variable) -> pd.DataFrame:
    # 到着した都市の順序を表示
    n = len(list_locations)
    list_result = []
    for i in range(1, n + 1):
        list_temp = []
        for j in range(1, n + 1):
            value = 0
            for t in range(len(list_cost)):
                tmp = pulp.value(opt_var.x[i, j, t])
                if tmp is not None:
                    value += tmp
            list_temp.append(value)
        list_result.append(list_temp)

    df = pd.DataFrame(list_result)
    df.columns = list_locations
    df.index = list_locations
    return df


def _get_time_matrix(list_cost: list[np.ndarray], list_locations: list[str], opt_var: Variable) -> pd.DataFrame:
    n = len(list_locations)
    # 到着した都市の順序を表示
    list_result = []
    for i in range(1, n + 1):
        list_temp = []
        for t in range(len(list_cost)):
            value = 0
            for j in range(1, n + 1):
                tmp = pulp.value(opt_var.x[j, i, t])
                if tmp is not None:
                    value += tmp
            # if value is not None and value > 0:
            list_temp.append(value)
        list_result.append(list_temp)
    df = pd.DataFrame(list_result)
    return df


def _get_plan(cost: CostMatrix, df_time: pd.DataFrame, list_locations: list[str], opt_var: Variable, visit_time, return_time) -> pd.DataFrame:
    # 到着した都市の順序を表示
    key_arrive = "到着時刻"
    key_leave = "出発時刻"
    key_attraction = "施設名"
    key_wait = "予測待ち時間"
    key_move = "移動時間"
    dict_loc2time = {}
    for i in range(df_time.shape[1]):
        idx = df_time[df_time.iloc[:, i] == 1].index
        if len(idx) > 0:
            for j in idx:
                dict_loc2time[list_locations[j]] = i
    n = len(list_locations)
    dict_result = {key_attraction: [], key_arrive: [], key_leave: [], key_wait: [], key_move: []}
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            tmp = pulp.value(opt_var.u[i, j])
            if tmp > 0:
                attraction = list_locations[i - 1]
                k = 0
                while k == 0 or k == i - 1:
                    k += 1
                dict_result[key_wait].append(cost.list_wait[dict_loc2time[attraction]][k][i - 1])
                dict_result[key_move].append(cost.list_move[dict_loc2time[attraction]][i - 1][j - 1])
                dict_result[key_attraction].append(attraction)
                dict_result[key_leave].append(visit_time + timedelta(seconds=tmp))
                dict_result[key_arrive].append(visit_time + timedelta(seconds=tmp))
    # 入園
    dict_result[key_attraction].append("入園")
    dict_result[key_leave].append(visit_time)
    dict_result[key_arrive].append(visit_time)
    dict_result[key_wait].append(0)
    dict_result[key_move].append(np.nan)

    # 退園
    dict_result[key_attraction].append("退園")
    dict_result[key_leave].append(return_time)
    dict_result[key_arrive].append(return_time)
    dict_result[key_wait].append(0)
    dict_result[key_move].append(0)
    df = pd.DataFrame(dict_result)
    df = df.sort_values(key_leave).reset_index(drop=True)

    dict_attraction2idx = {attraction: idx for idx, attraction in enumerate(sorted(list_locations))}
    first_idx = dict_attraction2idx[df.iloc[1][key_attraction]]
    first_move = cost.list_move[0][0][first_idx]
    df.loc[0, key_move] = first_move
    df[key_move] = df[key_move].shift(1).fillna(0)
    df["pre_" + key_leave] = df[key_leave].shift(1).fillna(0)
    df[key_arrive] = df.apply(lambda x: pd.to_datetime(x["pre_" + key_leave]) + timedelta(seconds=x[key_move]), axis=1)
    df = df.drop("pre_" + key_leave, axis=1)
    df[key_arrive] = df[key_arrive].dt.time
    df[key_leave] = df[key_leave].dt.time
    df[key_wait] = round(df[key_wait] / 60)
    df[key_move] = round(df[key_move] / 60)
    return df
