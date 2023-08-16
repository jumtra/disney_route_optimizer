from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class ConfigManeger:

    """config管理クラス"""

    config: DictConfig[str, Any]
    config_dir: str

    @classmethod
    def from_yaml(
        cls,
        config_dir: str,
        config_wp_yaml_path: str,
        config_opt_yaml_path: str,
        common_yaml_path: str,
        enable_merge_cli_args: bool = False,
    ) -> ConfigManeger:
        """yamlファイルの読み込み
        enable_merge_cli_args: CLI引数からconfigを受け取る場合使用
        """

        wp_config = OmegaConf.load(f"{config_dir}/{config_wp_yaml_path}")
        opt_config = OmegaConf.load(f"{config_dir}/{config_opt_yaml_path}")
        yaml_config = OmegaConf.merge(wp_config, opt_config)
        common_config = OmegaConf.load(f"{config_dir}/{common_yaml_path}")
        config = OmegaConf.merge(common_config, yaml_config)
        if not enable_merge_cli_args:
            # pythonの辞書に変換してオブジェクトにする
            return cls(config, config_dir)

        cli_config = OmegaConf.from_cli()
        config = OmegaConf.merge(config, cli_config)
        return cls(config, config_dir)

    # NOTE: 操作がいるconfigだけ定義
    def save_yaml(self, filename: Union[str, Path]) -> None:
        """設定をyamlファイルとしてファイル出力する"""

        with open(filename, "w") as f:
            f.write(OmegaConf.to_yaml(self.config, resolve=True))
