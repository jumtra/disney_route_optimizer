from datetime import timedelta

import pandas as pd
import pulp

from disney_route_optimize.common.config_maneger import ConfigManeger
from disney_route_optimize.route_optimize.dataclass.do_cost import CostMatrix
from disney_route_optimize.route_optimize.model.get_constraint import get_constraint
from disney_route_optimize.route_optimize.model.get_objective import get_objective
from disney_route_optimize.route_optimize.model.variable import Variable


def optimize(config_maneger: ConfigManeger, cost: CostMatrix):
    list_cost = cost.list_cost
    list_locations = cost.list_target_cols
    model = pulp.LpProblem("ParkTSP", pulp.LpMaximize)
    opt_var = Variable(cost_list=list_cost, len_times=len(list_cost), len_locations=len(list_locations))
    return_time = pd.to_datetime(config_maneger.config.common.return_time)
    visit_time = pd.to_datetime(config_maneger.config.common.visit_time)
    global_max_time = (return_time - visit_time) / timedelta(days=1) * 24 * 60
    sep_time = config_maneger.config.cost.sep_time
    for const_name, is_use in config_maneger.config.first_opt.constraints.items():
        if is_use:
            model = get_constraint(constraint_name=const_name)(
                model=model, variable=opt_var, sep_time=sep_time, global_time=global_max_time
            ).get_model()
    for objective_name, is_use in config_maneger.config.first_opt.objective.items():
        if is_use:
            model = get_objective(objective_name=objective_name)(model=model, variable=opt_var).get_model()
    model.solve(pulp.PULP_CBC_CMD(threads=16, timeLimit=20))

    ans = {}
    n = len(list_locations)
    # if pulp.LpStatus[model.status] in ["Optimal",:
    # 結果を表示
    print(pulp.LpStatus[model.status])
    # print(f"最適解が見つかりました:Max{n}")
    for i in range(n):
        for j in range(n):
            value = 0
            for t_i in range(len(list_cost)):
                tmp = pulp.value(opt_var.x[i + 1, j + 1, t_i])
                if tmp is not None:
                    value += tmp
            if value == 1:
                print(f"{list_locations[i]} → {list_locations[j]}")
                ans[i] = j
    # 到着した都市の順序を表示
    list_temp = []
    for i in range(1, n + 1):
        lis = []
        for j in range(1, n + 1):
            value = 0
            for t_i in range(len(list_cost)):
                tmp = pulp.value(opt_var.x[i, j, t_i])
                if tmp is not None:
                    value += tmp
            # if value is not None and value > 0:
            lis.append(value)
        list_temp.append(lis)

    df = pd.DataFrame(list_temp)

    # 到着した都市の順序を表示
    list_temp = []
    for i in range(1, n + 1):
        lis = []
        for t_i in range(len(list_cost)):
            value = 0
            for j in range(1, n + 1):
                tmp = pulp.value(opt_var.x[i, j, t_i])
                if tmp is not None:
                    value += tmp
            # if value is not None and value > 0:
            lis.append(value)
        list_temp.append(lis)
    df = pd.DataFrame(list_temp)
    for i in range(df.shape[1]):
        idx = df[df.iloc[:, i] == 1].index
        if len(idx) > 0:
            for j in idx:
                print(f"Time_{i} : Visit{list_locations[j]}")

    list_temp = []
    for i in range(1, n + 1):
        lis = []
        value = 0
        for j in range(1, n + 1):
            tmp = pulp.value(opt_var.u[i, j])
            if tmp > 0:
                print(f"{list_locations[i]}_{list_locations[j]} TIME:{tmp}")
    obj_num = pulp.value(model.objective)
    # else:
    #    print("最適解が見つかりませんでした")
