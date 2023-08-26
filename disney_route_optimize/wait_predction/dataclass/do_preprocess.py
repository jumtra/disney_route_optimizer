import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from disney_route_optimize.common.config_manager import ConfigManager
from disney_route_optimize.wait_predction.preprocess.clean_data.make_clean_data import make_clean_data

logger = logging.getLogger(__name__)


@dataclass
class Preprocess:
    """前処理結果を保持するデータクラス"""

    config_manager: ConfigManager | None = None

    def __post_init__(self):
        path_weather = Path(self.config_manager.config.output.wp_output.weather_file)
        path_waittime = Path(self.config_manager.config.output.wp_output.waittime_file)
        if self.config_manager.config.tasks.wp_task.do_make_clean or (not path_waittime.exists()) or (not path_weather.exists()):
            logger.info("データ前処理")
            Path(self.config_manager.config.output.wp_output.path_preprocessed_dir).mkdir(parents=True, exist_ok=True)

            self.df_weather, self.df_waittime = make_clean_data(self.config_manager)

            self.df_weather.to_csv(path_weather, index=False)
            self.df_waittime.to_csv(path_waittime, index=False)
        else:
            logger.info("データ前処理をskip")
        self.df_weather = pd.read_csv(path_weather)
        self.df_waittime = pd.read_csv(path_waittime)
