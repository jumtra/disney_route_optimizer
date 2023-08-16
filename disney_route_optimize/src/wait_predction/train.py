import logging
from pathlib import Path

from disney_route_optimize.common.config_maneger import ConfigManeger
from disney_route_optimize.wait_predction.dataclass.do_feature import Features
from disney_route_optimize.wait_predction.model.lightgbm import LGBM

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config_maneger: ConfigManeger, feature: Features):
        self.config_maneger: ConfigManeger = config_maneger
        self.feature: Features = feature
        self.model = LGBM(config_maneger)

    def train(self) -> None:
        model_path = Path(self.config_maneger.config.output.wp_output.path_model)
        if self.config_maneger.config.tasks.wp_task.do_train or not model_path.exists():
            Path(self.config_maneger.config.output.wp_output.path_train_dir).mkdir(
                exist_ok=True, parents=True
            )

            dict_train = self.feature.dict_train
            dict_valid = self.feature.dict_valid
            self.model.train(dict_train=dict_train, dict_valid=dict_valid)
            self.model.save_model(model_path)

        else:
            logger.info("学習をskip")
