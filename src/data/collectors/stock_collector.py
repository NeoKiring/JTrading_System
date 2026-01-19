"""
Stock Data Collector Module
yfinanceを使用した株価データ収集
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from .base_collector import BaseCollector
from ..storage.database import get_db_manager
from ...utils.config import get_config
from ...utils.validators import validate_symbol, validate_date_range


class StockCollector(BaseCollector):
    """株価データ収集クラス"""

    def __init__(self):
        """初期化"""
        super().__init__(name="StockCollector")
        self.db_manager = get_db_manager()
        self.default_start_date = get_config('data_collection.stock.start_date', '2014-01-01')
        self.default_interval = get_config('data_collection.stock.interval', '1d')

    def collect(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: Optional[str] = None,
        save_to_db: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        株価データを収集

        Args:
            symbol: 銘柄コード（例: "7203.T"）
            start_date: 開始日（YYYY-MM-DD）
            end_date: 終了日（YYYY-MM-DD）
            interval: データ間隔（1d, 1h, 5m等）
            save_to_db: データベースに保存するか

        Returns:
            pd.DataFrame: 株価データ
        """
        try:
            # バリデーション
            validate_symbol(symbol)

            # デフォルト値の設定
            if start_date is None:
                start_date = self.default_start_date
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if interval is None:
                interval = self.default_interval

            # 日付範囲の検証
            validate_date_range(start_date, end_date)

            self._log_collection_start(f"{symbol} ({start_date} to {end_date})")

            # yfinanceでデータ取得
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)

            if df.empty:
                self.logger.warning(f"No data retrieved for {symbol}")
                return None

            # データの前処理
            df = self._preprocess_data(df, symbol)

            self._log_collection_complete(symbol, len(df))

            # データベースに保存
            if save_to_db:
                self._save_to_database(df, symbol)

            return df

        except Exception as e:
            self._handle_error(e, f"symbol={symbol}")
            return None

    def collect_multiple(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: Optional[str] = None,
        save_to_db: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        複数銘柄の株価データを収集

        Args:
            symbols: 銘柄コードのリスト
            start_date: 開始日
            end_date: 終了日
            interval: データ間隔
            save_to_db: データベースに保存するか

        Returns:
            Dict[str, pd.DataFrame]: 銘柄コードをキーとした株価データの辞書
        """
        self.logger.info(f"Collecting data for {len(symbols)} symbols")

        results = {}
        for symbol in symbols:
            df = self.collect(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                save_to_db=save_to_db
            )
            if df is not None:
                results[symbol] = df

        self.logger.info(f"Successfully collected data for {len(results)}/{len(symbols)} symbols")
        return results

    def update_recent_data(
        self,
        symbol: str,
        days: int = 7,
        save_to_db: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        最近N日間のデータを更新

        Args:
            symbol: 銘柄コード
            days: 取得する日数
            save_to_db: データベースに保存するか

        Returns:
            pd.DataFrame: 株価データ
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        return self.collect(
            symbol=symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            save_to_db=save_to_db
        )

    def _preprocess_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        データの前処理

        Args:
            df: 生データ
            symbol: 銘柄コード

        Returns:
            pd.DataFrame: 前処理済みデータ
        """
        # カラム名を標準化
        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        # 銘柄コードを追加
        df['symbol'] = symbol

        # 不要なカラムを削除
        columns_to_keep = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
        df = df[[col for col in columns_to_keep if col in df.columns]]

        # 欠損値の処理
        df = df.dropna(subset=['close'])

        # 日付型に変換
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date

        return df

    def _save_to_database(self, df: pd.DataFrame, symbol: str):
        """
        データベースに保存

        Args:
            df: 株価データ
            symbol: 銘柄コード
        """
        try:
            # DataFrameを辞書のリストに変換
            records = df.to_dict('records')

            # データベースに保存
            self.db_manager.save_stock_prices(records)

            self.logger.info(f"Saved {len(records)} records to database for {symbol}")

        except Exception as e:
            self.logger.error(f"Failed to save data to database: {e}")

    def get_latest_date(self, symbol: str) -> Optional[datetime]:
        """
        データベース内の最新日付を取得

        Args:
            symbol: 銘柄コード

        Returns:
            datetime: 最新日付
        """
        try:
            prices = self.db_manager.get_stock_prices(symbol)
            if prices:
                latest = max(prices, key=lambda x: x.date)
                return datetime.combine(latest.date, datetime.min.time())
            return None

        except Exception as e:
            self.logger.error(f"Failed to get latest date: {e}")
            return None

    def get_available_data_range(self, symbol: str) -> Optional[tuple]:
        """
        利用可能なデータ範囲を取得

        Args:
            symbol: 銘柄コード

        Returns:
            tuple: (開始日, 終了日)
        """
        try:
            prices = self.db_manager.get_stock_prices(symbol)
            if prices:
                dates = [p.date for p in prices]
                return (min(dates), max(dates))
            return None

        except Exception as e:
            self.logger.error(f"Failed to get data range: {e}")
            return None
