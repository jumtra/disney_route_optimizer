import logging
from pathlib import Path

from disney_route_optimize.common.config_manager import ConfigManager
from disney_route_optimize.wait_predction.dataclass.do_feature import Features
from disney_route_optimize.wait_predction.model.lightgbm import LGBM

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config_manager: ConfigManager, feature: Features):
        self.config_manager: ConfigManager = config_manager
        self.feature: Features = feature
        self.model = LGBM(config_manager)

    def train(self) -> None:
        model_path = Path(self.config_manager.config.output.wp_output.path_model)
        is_only_predict = self.config_manager.config.common.is_only_predict
        Path(self.config_manager.config.output.wp_output.path_train_dir).mkdir(exist_ok=True, parents=True)
        if (self.config_manager.config.tasks.wp_task.do_train or not model_path.exists()) and not is_only_predict:
            dict_train = self.feature.dict_train
            dict_valid = self.feature.dict_valid
            self.model.train(dict_train=dict_train, dict_valid=dict_valid)
            self.model.save_model(model_path)

        else:
            if is_only_predict:
                self.model.load_model(model_path)
            logger.info("学習をskip")
