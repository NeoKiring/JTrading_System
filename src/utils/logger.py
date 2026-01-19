"""
Logger Module
高度なログ管理機能を提供
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional


class LoggerManager:
    """ログ管理クラス（Singletonパターン）"""

    _instance: Optional['LoggerManager'] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初期化"""
        if not self._initialized:
            self._initialized = True
            self._setup_default_logger()

    def _setup_default_logger(self):
        """デフォルトロガーの設定"""
        # デフォルトのハンドラーを削除
        logger.remove()

        # コンソール出力の設定
        logger.add(
            sys.stdout,
            level="INFO",
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            colorize=True
        )

    def setup(
        self,
        log_level: str = "INFO",
        log_format: str = "[{time:YYYY-MM-DD HH:mm:ss}] [{level}] [{name}] {message}",
        rotation: str = "10 MB",
        retention: str = "30 days",
        app_log_path: str = "logs/app.log",
        error_log_path: str = "logs/error.log",
        trading_log_path: str = "logs/trading.log"
    ):
        """
        ロガーの詳細設定

        Args:
            log_level: ログレベル（DEBUG, INFO, WARNING, ERROR, CRITICAL）
            log_format: ログフォーマット
            rotation: ログローテーション設定
            retention: ログ保持期間
            app_log_path: アプリケーションログファイルパス
            error_log_path: エラーログファイルパス
            trading_log_path: トレーディングログファイルパス
        """
        # ログディレクトリの作成
        for log_path in [app_log_path, error_log_path, trading_log_path]:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)

        # デフォルトハンドラーを削除
        logger.remove()

        # コンソール出力（カラー付き）
        logger.add(
            sys.stdout,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
                   "<level>{message}</level>",
            colorize=True
        )

        # アプリケーションログファイル
        logger.add(
            app_log_path,
            level=log_level,
            format=log_format,
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
            enqueue=True  # スレッドセーフ
        )

        # エラーログファイル（ERROR以上のみ）
        logger.add(
            error_log_path,
            level="ERROR",
            format=log_format,
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
            enqueue=True,
            backtrace=True,  # スタックトレースを記録
            diagnose=True     # 詳細な診断情報
        )

        # トレーディングログファイル（取引関連のログ専用）
        logger.add(
            trading_log_path,
            level="INFO",
            format=log_format,
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
            enqueue=True,
            filter=lambda record: "trading" in record["extra"]
        )

        logger.info(f"Logger initialized with level: {log_level}")

    @staticmethod
    def get_logger(name: str = __name__):
        """
        ロガーインスタンスの取得

        Args:
            name: ロガー名

        Returns:
            logger: ロガーインスタンス
        """
        return logger.bind(name=name)

    @staticmethod
    def get_trading_logger(name: str = __name__):
        """
        トレーディングログ専用ロガーの取得

        Args:
            name: ロガー名

        Returns:
            logger: トレーディングロガーインスタンス
        """
        return logger.bind(name=name, trading=True)


# シングルトンインスタンス
_logger_manager = LoggerManager()


def setup_logger(
    log_level: str = "INFO",
    log_format: str = "[{time:YYYY-MM-DD HH:mm:ss}] [{level}] [{name}] {message}",
    rotation: str = "10 MB",
    retention: str = "30 days",
    app_log_path: str = "logs/app.log",
    error_log_path: str = "logs/error.log",
    trading_log_path: str = "logs/trading.log"
):
    """
    ロガーのセットアップ（グローバル関数）

    Args:
        log_level: ログレベル
        log_format: ログフォーマット
        rotation: ログローテーション設定
        retention: ログ保持期間
        app_log_path: アプリケーションログファイルパス
        error_log_path: エラーログファイルパス
        trading_log_path: トレーディングログファイルパス
    """
    _logger_manager.setup(
        log_level=log_level,
        log_format=log_format,
        rotation=rotation,
        retention=retention,
        app_log_path=app_log_path,
        error_log_path=error_log_path,
        trading_log_path=trading_log_path
    )


def get_logger(name: str = __name__):
    """
    ロガーインスタンスの取得（グローバル関数）

    Args:
        name: ロガー名

    Returns:
        logger: ロガーインスタンス
    """
    return _logger_manager.get_logger(name)


def get_trading_logger(name: str = __name__):
    """
    トレーディングログ専用ロガーの取得（グローバル関数）

    Args:
        name: ロガー名

    Returns:
        logger: トレーディングロガーインスタンス
    """
    return _logger_manager.get_trading_logger(name)
