import logging
from pathlib import Path

from .scraper.distance_scraper import get_distance_data
from .scraper.wait_time_scraper import get_wait_time_data
from .scraper.weather_scraper import get_weather_data

logger = logging.getLogger(__name__)


def scraping(path_file: Path) -> None:
    logger.info("データ取得開始")
    get_weather_data(path_file / "weather")
    get_wait_time_data(path_file, is_land=True)
    get_wait_time_data(path_file, is_land=False)
    get_distance_data(path_file)
    logger.info("データ取得終了")
