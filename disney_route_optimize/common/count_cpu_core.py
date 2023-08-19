import multiprocessing
from typing import Literal


def get_cpu_core(level: Literal["high", "middle", "low"]) -> int:
    """並列化で使用するcpuコア数を取得

    - high: 全コア
    - middle: 3/4コア
    - low: 1/2コア

    """
    core = multiprocessing.cpu_count()
    if level == "middle":
        return int(core / 4 * 3)

    if level == "low":
        return int(core / 2)

    return core
