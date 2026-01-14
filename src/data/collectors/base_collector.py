"""
Base Collector Module
データ収集の基底クラス
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from datetime import datetime
from ...utils.logger import get_logger

logger = get_logger(__name__)


class BaseCollector(ABC):
    """データ収集の基底クラス"""

    def __init__(self, name: str = "BaseCollector"):
        """
        初期化

        Args:
            name: コレクター名
        """
        self.name = name
        self.logger = get_logger(name)

    @abstractmethod
    def collect(self, *args, **kwargs) -> Any:
        """
        データ収集の抽象メソッド

        Returns:
            収集したデータ
        """
        pass

    def _validate_date_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> bool:
        """
        日付範囲の妥当性を検証

        Args:
            start_date: 開始日
            end_date: 終了日

        Returns:
            bool: 妥当性
        """
        if start_date >= end_date:
            self.logger.error(f"Invalid date range: {start_date} >= {end_date}")
            return False

        if end_date > datetime.now():
            self.logger.warning(f"End date is in the future: {end_date}")

        return True

    def _handle_error(self, error: Exception, context: str = ""):
        """
        エラーハンドリング

        Args:
            error: エラー
            context: エラーコンテキスト
        """
        error_msg = f"{self.name} error"
        if context:
            error_msg += f" ({context})"
        error_msg += f": {str(error)}"

        self.logger.error(error_msg)

    def _log_collection_start(self, target: str):
        """
        収集開始のログ

        Args:
            target: 収集対象
        """
        self.logger.info(f"Starting data collection for {target}")

    def _log_collection_complete(self, target: str, count: int):
        """
        収集完了のログ

        Args:
            target: 収集対象
            count: 収集件数
        """
        self.logger.info(f"Completed data collection for {target}: {count} records")
