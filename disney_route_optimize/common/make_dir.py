import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def make_dir(output_path: Path, is_del=True) -> None:
    if output_path.exist() and is_del:
        shutil.rmtree(output_path)
        logger.info(f"{output_path}を削除しました")
    output_path.mkdir(exist_ok=True, parents=True)
