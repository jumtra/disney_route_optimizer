import tempfile

import lightgbm as lgb


def get_bestiter_model(model: lgb.Booster) -> lgb.Booster:
    """best_iterationのmodelを返す"""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=True) as tmp:
        model.save_model(tmp.name, num_iteration=model.best_iteration)
        tmp_model_path = tmp.name
        return lgb.Booster(model_file=tmp_model_path)
