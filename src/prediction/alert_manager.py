"""
Alert Manager Module
アラート・通知管理モジュール
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum
import json
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.config import get_config

logger = get_logger(__name__)


class AlertLevel(Enum):
    """アラートレベル"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    """アラートタイプ"""
    PRICE_UP = "price_up"  # 価格上昇予測
    PRICE_DOWN = "price_down"  # 価格下降予測
    HIGH_VOLATILITY = "high_volatility"  # 高ボラティリティ
    MODEL_ERROR = "model_error"  # モデルエラー
    DATA_ERROR = "data_error"  # データエラー
    CONFIDENCE_LOW = "confidence_low"  # 信頼度低下


class Alert:
    """アラート情報"""

    def __init__(
        self,
        alert_type: AlertType,
        level: AlertLevel,
        symbol: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        初期化

        Args:
            alert_type: アラートタイプ
            level: アラートレベル
            symbol: 銘柄コード
            message: メッセージ
            data: 追加データ
            timestamp: タイムスタンプ
        """
        self.alert_type = alert_type
        self.level = level
        self.symbol = symbol
        self.message = message
        self.data = data or {}
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'alert_type': self.alert_type.value,
            'level': self.level.value,
            'symbol': self.symbol,
            'message': self.message,
            'data': self.data,
            'timestamp': self.timestamp.isoformat()
        }

    def __str__(self) -> str:
        return (f"[{self.level.value.upper()}] {self.symbol}: "
                f"{self.message} ({self.timestamp.strftime('%Y-%m-%d %H:%M:%S')})")


class AlertManager:
    """
    アラート管理クラス

    予測結果を監視し、条件に応じてアラートを生成・通知します。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初期化

        Args:
            config: アラート設定（省略時は設定ファイルから取得）
        """
        self.config = config or get_config('alerts', {})

        # アラート履歴
        self.alert_history: List[Alert] = []
        self.max_history = self.config.get('max_history', 1000)

        # 通知コールバック
        self.notification_callbacks: List[Callable] = []

        # アラート条件
        self.thresholds = {
            'price_change_up': self.config.get('price_change_up_threshold', 5.0),  # %
            'price_change_down': self.config.get('price_change_down_threshold', -5.0),  # %
            'high_volatility': self.config.get('high_volatility_threshold', 10.0),  # %
            'confidence_low': self.config.get('confidence_low_threshold', 0.3)  # 信頼区間幅/予測価格
        }

    def check_prediction(self, prediction: Dict[str, Any]) -> List[Alert]:
        """
        予測結果をチェックしてアラートを生成

        Args:
            prediction: 予測結果

        Returns:
            List[Alert]: 生成されたアラートリスト
        """
        alerts = []

        # エラーチェック
        if prediction.get('status') == 'error':
            alert = Alert(
                alert_type=AlertType.MODEL_ERROR,
                level=AlertLevel.WARNING,
                symbol=prediction['symbol'],
                message=f"予測エラー: {prediction.get('error')}",
                data=prediction
            )
            alerts.append(alert)
            self._notify(alert)
            return alerts

        symbol = prediction['symbol']
        change_pct = prediction.get('predicted_change_pct', 0)

        # 価格上昇アラート
        if change_pct >= self.thresholds['price_change_up']:
            level = AlertLevel.CRITICAL if change_pct >= 10.0 else AlertLevel.WARNING
            alert = Alert(
                alert_type=AlertType.PRICE_UP,
                level=level,
                symbol=symbol,
                message=f"大幅な価格上昇が予測されました: {change_pct:+.2f}%",
                data=prediction
            )
            alerts.append(alert)
            self._notify(alert)

        # 価格下降アラート
        elif change_pct <= self.thresholds['price_change_down']:
            level = AlertLevel.CRITICAL if change_pct <= -10.0 else AlertLevel.WARNING
            alert = Alert(
                alert_type=AlertType.PRICE_DOWN,
                level=level,
                symbol=symbol,
                message=f"大幅な価格下降が予測されました: {change_pct:+.2f}%",
                data=prediction
            )
            alerts.append(alert)
            self._notify(alert)

        # 高ボラティリティアラート
        if self._check_high_volatility(prediction):
            alert = Alert(
                alert_type=AlertType.HIGH_VOLATILITY,
                level=AlertLevel.INFO,
                symbol=symbol,
                message=f"高いボラティリティが予測されています",
                data=prediction
            )
            alerts.append(alert)
            self._notify(alert)

        # 信頼度低下アラート
        if self._check_low_confidence(prediction):
            alert = Alert(
                alert_type=AlertType.CONFIDENCE_LOW,
                level=AlertLevel.INFO,
                symbol=symbol,
                message=f"予測の信頼度が低下しています",
                data=prediction
            )
            alerts.append(alert)
            self._notify(alert)

        # 履歴に保存
        for alert in alerts:
            self._add_to_history(alert)

        return alerts

    def _check_high_volatility(self, prediction: Dict[str, Any]) -> bool:
        """
        高ボラティリティをチェック

        Args:
            prediction: 予測結果

        Returns:
            bool: 高ボラティリティの場合True
        """
        predicted_price = prediction.get('predicted_price')
        confidence_upper = prediction.get('confidence_upper')
        confidence_lower = prediction.get('confidence_lower')

        if not all([predicted_price, confidence_upper, confidence_lower]):
            return False

        # 信頼区間の幅を計算
        volatility_pct = ((confidence_upper - confidence_lower) / predicted_price) * 100

        return volatility_pct > self.thresholds['high_volatility']

    def _check_low_confidence(self, prediction: Dict[str, Any]) -> bool:
        """
        低信頼度をチェック

        Args:
            prediction: 予測結果

        Returns:
            bool: 低信頼度の場合True
        """
        predicted_price = prediction.get('predicted_price')
        confidence_upper = prediction.get('confidence_upper')
        confidence_lower = prediction.get('confidence_lower')

        if not all([predicted_price, confidence_upper, confidence_lower]):
            return False

        # 信頼区間幅/予測価格
        confidence_ratio = (confidence_upper - confidence_lower) / predicted_price

        return confidence_ratio > self.thresholds['confidence_low']

    def _notify(self, alert: Alert):
        """
        アラートを通知

        Args:
            alert: アラート
        """
        # ログ出力
        logger_method = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.CRITICAL: logger.error
        }.get(alert.level, logger.info)

        logger_method(f"Alert: {alert}")

        # コールバック実行
        for callback in self.notification_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in notification callback: {e}")

    def _add_to_history(self, alert: Alert):
        """
        アラートを履歴に追加

        Args:
            alert: アラート
        """
        self.alert_history.append(alert)

        # 履歴サイズ制限
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]

    def register_callback(self, callback: Callable[[Alert], None]):
        """
        通知コールバックを登録

        Args:
            callback: コールバック関数（引数: Alert）
        """
        self.notification_callbacks.append(callback)

    def get_alerts(
        self,
        symbol: Optional[str] = None,
        level: Optional[AlertLevel] = None,
        alert_type: Optional[AlertType] = None,
        limit: int = 100
    ) -> List[Alert]:
        """
        アラート履歴を取得

        Args:
            symbol: 銘柄コード（フィルター）
            level: アラートレベル（フィルター）
            alert_type: アラートタイプ（フィルター）
            limit: 取得件数

        Returns:
            List[Alert]: アラートリスト
        """
        alerts = self.alert_history

        # フィルタリング
        if symbol:
            alerts = [a for a in alerts if a.symbol == symbol]

        if level:
            alerts = [a for a in alerts if a.level == level]

        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]

        # 最新順にソート
        alerts = sorted(alerts, key=lambda a: a.timestamp, reverse=True)

        return alerts[:limit]

    def clear_alerts(self, symbol: Optional[str] = None):
        """
        アラート履歴をクリア

        Args:
            symbol: 銘柄コード（指定時はその銘柄のみクリア）
        """
        if symbol:
            self.alert_history = [a for a in self.alert_history if a.symbol != symbol]
        else:
            self.alert_history = []

        logger.info(f"Alert history cleared{' for ' + symbol if symbol else ''}")

    def export_alerts(self, filepath: str, format: str = 'json'):
        """
        アラート履歴をエクスポート

        Args:
            filepath: 出力ファイルパス
            format: フォーマット（json or csv）
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            data = [alert.to_dict() for alert in self.alert_history]
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        elif format == 'csv':
            import pandas as pd
            data = [alert.to_dict() for alert in self.alert_history]
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False, encoding='utf-8')

        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Alerts exported to {filepath}")

    def get_summary(self) -> Dict[str, Any]:
        """
        アラートサマリーを取得

        Returns:
            Dict: サマリー情報
        """
        if not self.alert_history:
            return {
                'total': 0,
                'by_level': {},
                'by_type': {},
                'recent_count': 0
            }

        # レベル別集計
        by_level = {}
        for level in AlertLevel:
            count = sum(1 for a in self.alert_history if a.level == level)
            by_level[level.value] = count

        # タイプ別集計
        by_type = {}
        for alert_type in AlertType:
            count = sum(1 for a in self.alert_history if a.alert_type == alert_type)
            by_type[alert_type.value] = count

        # 最近1時間のアラート数
        one_hour_ago = datetime.now().timestamp() - 3600
        recent_count = sum(
            1 for a in self.alert_history
            if a.timestamp.timestamp() > one_hour_ago
        )

        return {
            'total': len(self.alert_history),
            'by_level': by_level,
            'by_type': by_type,
            'recent_count': recent_count,
            'latest': self.alert_history[-1].to_dict() if self.alert_history else None
        }
