from .objective.base_objective import BaseObjective
from .objective.objective_total_popular_visit import ObjectiveTotalPopularVisit
from .objective.objective_total_visit import ObjectiveTotalVisit

_dict_name_map_objective: dict[str, BaseObjective] = {
    ObjectiveTotalVisit.objective_name: ObjectiveTotalVisit,
    ObjectiveTotalPopularVisit.objective_name: ObjectiveTotalPopularVisit,
}


def get_objective(objective_name: str) -> BaseObjective:
    """目的関数名からObjective Classを取得する"""
    return _dict_name_map_objective[objective_name]
