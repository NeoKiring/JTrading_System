"""
Configuration Manager Module
設定ファイルの読み込みと管理
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from .logger import get_logger

logger = get_logger(__name__)


class ConfigManager:
    """設定管理クラス（Singletonパターン）"""

    _instance: Optional['ConfigManager'] = None
    _config: Dict[str, Any] = {}
    _symbols: Dict[str, Any] = {}
    _model_config: Dict[str, Any] = {}
    _api_keys: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初期化"""
        if not self._config:
            self.load_all_configs()

    def load_all_configs(
        self,
        config_dir: str = "config",
        config_file: str = "config.yaml",
        symbols_file: str = "symbols.yaml",
        model_config_file: str = "model_config.yaml",
        api_keys_file: str = "api_keys.yaml"
    ):
        """
        全ての設定ファイルを読み込む

        Args:
            config_dir: 設定ファイルディレクトリ
            config_file: メイン設定ファイル名
            symbols_file: 銘柄リストファイル名
            model_config_file: モデル設定ファイル名
            api_keys_file: APIキーファイル名
        """
        config_path = Path(config_dir)

        # メイン設定の読み込み
        self._config = self._load_yaml(config_path / config_file)
        logger.info(f"Loaded main config from {config_path / config_file}")

        # 銘柄リストの読み込み
        self._symbols = self._load_yaml(config_path / symbols_file)
        logger.info(f"Loaded symbols from {config_path / symbols_file}")

        # モデル設定の読み込み
        self._model_config = self._load_yaml(config_path / model_config_file)
        logger.info(f"Loaded model config from {config_path / model_config_file}")

        # APIキーの読み込み（オプショナル）
        api_keys_path = config_path / api_keys_file
        if api_keys_path.exists():
            self._api_keys = self._load_yaml(api_keys_path)
            logger.info(f"Loaded API keys from {api_keys_path}")
        else:
            logger.warning(f"API keys file not found: {api_keys_path}")
            self._api_keys = {}

    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """
        YAMLファイルを読み込む

        Args:
            file_path: YAMLファイルパス

        Returns:
            dict: 読み込んだ設定
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config if config else {}
        except FileNotFoundError:
            logger.error(f"Config file not found: {file_path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {file_path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error loading {file_path}: {e}")
            return {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        設定値を取得（ドット記法対応）

        Args:
            key: 設定キー（例: "app.name", "data_collection.stock.start_date"）
            default: デフォルト値

        Returns:
            設定値
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_symbols(self, list_name: Optional[str] = None) -> list:
        """
        銘柄リストを取得

        Args:
            list_name: リスト名（None の場合はアクティブリスト）

        Returns:
            list: 銘柄リスト
        """
        if list_name is None:
            list_name = self._symbols.get('active_list', 'test_symbols')

        return self._symbols.get(list_name, [])

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        モデル設定を取得

        Args:
            model_name: モデル名（例: "xgboost", "lightgbm"）

        Returns:
            dict: モデル設定
        """
        return self._model_config.get(model_name, {})

    def get_api_key(self, service: str, key_name: str = "api_key") -> Optional[str]:
        """
        APIキーを取得

        Args:
            service: サービス名（例: "newsapi", "alphavantage"）
            key_name: キー名（デフォルト: "api_key"）

        Returns:
            str: APIキー
        """
        service_config = self._api_keys.get(service, {})
        return service_config.get(key_name)

    @property
    def config(self) -> Dict[str, Any]:
        """メイン設定を取得"""
        return self._config

    @property
    def symbols(self) -> Dict[str, Any]:
        """銘柄設定を取得"""
        return self._symbols

    @property
    def model_config(self) -> Dict[str, Any]:
        """モデル設定を取得"""
        return self._model_config

    @property
    def api_keys(self) -> Dict[str, Any]:
        """APIキーを取得"""
        return self._api_keys


# シングルトンインスタンス
_config_manager = ConfigManager()


def get_config(key: str = None, default: Any = None) -> Any:
    """
    設定値を取得（グローバル関数）

    Args:
        key: 設定キー（None の場合は全設定を返す）
        default: デフォルト値

    Returns:
        設定値
    """
    if key is None:
        return _config_manager.config
    return _config_manager.get(key, default)


def get_symbols(list_name: Optional[str] = None) -> list:
    """
    銘柄リストを取得（グローバル関数）

    Args:
        list_name: リスト名

    Returns:
        list: 銘柄リスト
    """
    return _config_manager.get_symbols(list_name)


def get_all_symbols() -> Dict[str, Any]:
    """
    すべての銘柄リストを取得（グローバル関数）

    Returns:
        dict: 銘柄リスト設定
    """
    return _config_manager.symbols


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    モデル設定を取得（グローバル関数）

    Args:
        model_name: モデル名

    Returns:
        dict: モデル設定
    """
    return _config_manager.get_model_config(model_name)


def get_api_key(service: str, key_name: str = "api_key") -> Optional[str]:
    """
    APIキーを取得（グローバル関数）

    Args:
        service: サービス名
        key_name: キー名

    Returns:
        str: APIキー
    """
    return _config_manager.get_api_key(service, key_name)
