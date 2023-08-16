from .constraints.base_constraints import BaseConstraint
from .constraints.constraint_global_mtz import ConstraintGlobalMTZ
from .constraints.constraint_global_once import ConstraintGlobalOnce
from .constraints.constraint_global_time import ConstraintGlobalTime
from .constraints.constraint_mtz import ConstraintMTZ
from .constraints.constraint_play_time import ConstraintPlayTime
from .constraints.constraint_time_window import ConstraintTimeWindow
from .constraints.constraint_visit_once import ConstraintVisitOnce

_dict_name_map_constraint: dict[str, BaseConstraint] = {
    ConstraintVisitOnce.constraint_name: ConstraintVisitOnce,
    ConstraintGlobalTime.constraint_name: ConstraintGlobalTime,
    ConstraintGlobalMTZ.constraint_name: ConstraintGlobalMTZ,
    ConstraintGlobalOnce.constraint_name: ConstraintGlobalOnce,
    ConstraintMTZ.constraint_name: ConstraintMTZ,
    ConstraintPlayTime.constraint_name: ConstraintPlayTime,
    ConstraintTimeWindow.constraint_name: ConstraintTimeWindow,
}


def get_constraint(constraint_name: str) -> BaseConstraint:
    """制約名からConstraint Classを取得する"""
    return _dict_name_map_constraint[constraint_name]
