"""
Prediction Scheduler Module
予測スケジューラー
"""

import schedule
import time
import threading
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, time as dt_time

from .realtime_predictor import RealTimePredictor
from ..utils.logger import get_logger
from ..utils.config import get_config, get_symbols

logger = get_logger(__name__)


class PredictionScheduler:
    """
    予測スケジューラークラス

    定期的に予測を実行し、結果を記録します。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初期化

        Args:
            config: スケジューラー設定（省略時は設定ファイルから取得）
        """
        self.config = config or get_config('realtime_prediction', {})
        self.predictor = RealTimePredictor()

        self.is_running = False
        self.scheduler_thread = None

        # コールバック
        self.on_prediction_complete: Optional[Callable] = None
        self.on_batch_complete: Optional[Callable] = None
        self.on_error: Optional[Callable] = None

    def start(
        self,
        symbols: Optional[List[str]] = None,
        interval_minutes: Optional[int] = None,
        schedule_time: Optional[str] = None
    ):
        """
        スケジューラーを開始

        Args:
            symbols: 監視する銘柄リスト（省略時は設定から取得）
            interval_minutes: 実行間隔（分）
            schedule_time: 実行時刻（HH:MM形式、interval_minutesより優先）
        """
        if self.is_running:
            logger.warning("Scheduler is already running")
            return

        # 銘柄リスト
        if symbols is None:
            symbols = self._get_default_symbols()

        # スケジュール設定
        interval_minutes = interval_minutes or self.config.get('interval_minutes', 60)
        schedule_time = schedule_time or self.config.get('schedule_time')

        logger.info(f"Starting prediction scheduler for {len(symbols)} symbols")

        if schedule_time:
            logger.info(f"Scheduled to run daily at {schedule_time}")
            schedule.every().day.at(schedule_time).do(
                self._run_predictions, symbols=symbols
            )
        else:
            logger.info(f"Scheduled to run every {interval_minutes} minutes")
            schedule.every(interval_minutes).minutes.do(
                self._run_predictions, symbols=symbols
            )

        # スケジューラーをバックグラウンドで実行
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()

        logger.info("Prediction scheduler started")

    def stop(self):
        """スケジューラーを停止"""
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return

        logger.info("Stopping prediction scheduler...")
        self.is_running = False

        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)

        schedule.clear()
        logger.info("Prediction scheduler stopped")

    def run_now(self, symbols: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        即座に予測を実行

        Args:
            symbols: 予測する銘柄リスト（省略時はデフォルト）

        Returns:
            Dict: 予測結果
        """
        if symbols is None:
            symbols = self._get_default_symbols()

        logger.info(f"Running predictions now for {len(symbols)} symbols")
        return self._run_predictions(symbols)

    def _scheduler_loop(self):
        """スケジューラーループ（バックグラウンドスレッド）"""
        logger.info("Scheduler loop started")

        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                if self.on_error:
                    self.on_error(e)
                time.sleep(60)  # エラー時は1分待機

        logger.info("Scheduler loop stopped")

    def _run_predictions(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        予測を実行

        Args:
            symbols: 銘柄リスト

        Returns:
            Dict: 予測結果
        """
        logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                   f"Running batch prediction for {len(symbols)} symbols")

        start_time = time.time()

        try:
            # バッチ予測実行
            results = self.predictor.batch_predict(symbols, parallel=False)

            # 成功/失敗の集計
            success_count = sum(1 for r in results.values() if r['status'] == 'success')
            error_count = len(results) - success_count

            elapsed = time.time() - start_time

            logger.info(f"Batch prediction completed: "
                       f"{success_count} succeeded, {error_count} failed, "
                       f"elapsed={elapsed:.1f}s")

            # 各予測結果のコールバック
            if self.on_prediction_complete:
                for symbol, result in results.items():
                    self.on_prediction_complete(symbol, result)

            # バッチ完了コールバック
            if self.on_batch_complete:
                self.on_batch_complete(results)

            return results

        except Exception as e:
            logger.error(f"Error running batch predictions: {e}")
            if self.on_error:
                self.on_error(e)
            return {}

    def _get_default_symbols(self) -> List[str]:
        """
        デフォルトの銘柄リストを取得

        Returns:
            List[str]: 銘柄コードリスト
        """
        try:
            # 設定から銘柄リスト取得
            symbols = get_symbols('priority')

            if not symbols:
                logger.warning("No symbols found in config, using default")
                return ['7203.T']  # トヨタ自動車

            # dictのリストの場合、symbolキーを抽出
            if symbols and isinstance(symbols[0], dict):
                return [s['symbol'] for s in symbols]

            return symbols

        except Exception as e:
            logger.error(f"Error loading symbols: {e}")
            return ['7203.T']

    def set_callbacks(
        self,
        on_prediction_complete: Optional[Callable] = None,
        on_batch_complete: Optional[Callable] = None,
        on_error: Optional[Callable] = None
    ):
        """
        コールバック関数を設定

        Args:
            on_prediction_complete: 予測完了時のコールバック(symbol, result)
            on_batch_complete: バッチ完了時のコールバック(results)
            on_error: エラー発生時のコールバック(error)
        """
        self.on_prediction_complete = on_prediction_complete
        self.on_batch_complete = on_batch_complete
        self.on_error = on_error

    def get_next_run_time(self) -> Optional[datetime]:
        """
        次回実行予定時刻を取得

        Returns:
            datetime: 次回実行時刻（スケジュール未設定の場合はNone）
        """
        jobs = schedule.get_jobs()
        if not jobs:
            return None

        next_run = min(job.next_run for job in jobs if job.next_run)
        return next_run

    def is_market_open(self) -> bool:
        """
        市場が開いているかチェック

        日本市場: 平日9:00-15:00（昼休み11:30-12:30）

        Returns:
            bool: 市場が開いている場合True
        """
        now = datetime.now()

        # 土日チェック
        if now.weekday() >= 5:  # 5=土曜, 6=日曜
            return False

        # 時刻チェック
        current_time = now.time()

        # 午前（9:00-11:30）
        if dt_time(9, 0) <= current_time < dt_time(11, 30):
            return True

        # 午後（12:30-15:00）
        if dt_time(12, 30) <= current_time < dt_time(15, 0):
            return True

        return False


class SmartScheduler(PredictionScheduler):
    """
    スマートスケジューラー

    市場の開閉や取引量を考慮した最適なタイミングで予測を実行
    """

    def start(
        self,
        symbols: Optional[List[str]] = None,
        interval_minutes: Optional[int] = None,
        market_hours_only: bool = True
    ):
        """
        スマートスケジューラーを開始

        Args:
            symbols: 監視する銘柄リスト
            interval_minutes: 実行間隔（分）
            market_hours_only: 市場時間中のみ実行するか
        """
        self.market_hours_only = market_hours_only

        # 推奨スケジュール
        # 1. 寄付後: 9:15（寄付後の価格確定後）
        # 2. 前引前: 11:15（午前の動き反映）
        # 3. 後場始値後: 12:45（後場開始後）
        # 4. 引け後: 15:15（当日の全データ反映）

        if symbols is None:
            symbols = self._get_default_symbols()

        logger.info(f"Starting smart prediction scheduler for {len(symbols)} symbols")
        logger.info("Scheduled times: 09:15, 11:15, 12:45, 15:15")

        # 4つの時間帯でスケジュール
        schedule.every().day.at("09:15").do(self._smart_run, symbols=symbols, tag="morning_open")
        schedule.every().day.at("11:15").do(self._smart_run, symbols=symbols, tag="morning_close")
        schedule.every().day.at("12:45").do(self._smart_run, symbols=symbols, tag="afternoon_open")
        schedule.every().day.at("15:15").do(self._smart_run, symbols=symbols, tag="market_close")

        # スケジューラーをバックグラウンドで実行
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()

        logger.info("Smart prediction scheduler started")

    def _smart_run(self, symbols: List[str], tag: str) -> Dict[str, Dict[str, Any]]:
        """
        スマート実行（市場時間チェック付き）

        Args:
            symbols: 銘柄リスト
            tag: 実行タグ

        Returns:
            Dict: 予測結果
        """
        # 市場時間チェック
        if self.market_hours_only and tag != "market_close":
            # 引け後以外は市場開閉をチェック
            if datetime.now().weekday() >= 5:  # 土日はスキップ
                logger.info(f"Skipping {tag} prediction (weekend)")
                return {}

        logger.info(f"Running {tag} prediction")
        return self._run_predictions(symbols)
