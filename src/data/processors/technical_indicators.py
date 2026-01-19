"""
Technical Indicators Module
テクニカル指標計算（50種類以上）
"""

import pandas as pd
import numpy as np
from typing import Optional
from ...utils.logger import get_logger

logger = get_logger(__name__)


class TechnicalIndicators:
    """テクニカル指標計算クラス"""

    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        全ての主要テクニカル指標を追加

        Args:
            df: 株価データ（OHLCV）

        Returns:
            pd.DataFrame: 指標が追加されたデータ
        """
        df = df.copy()

        try:
            # トレンド系指標
            df = TechnicalIndicators.add_sma(df, periods=[5, 10, 20, 50, 200])
            df = TechnicalIndicators.add_ema(df, periods=[5, 10, 20, 50, 200])
            df = TechnicalIndicators.add_wma(df, period=20)
            df = TechnicalIndicators.add_macd(df)
            df = TechnicalIndicators.add_bollinger_bands(df)

            # モメンタム系指標
            df = TechnicalIndicators.add_rsi(df, periods=[14, 28])
            df = TechnicalIndicators.add_stochastic(df)
            df = TechnicalIndicators.add_cci(df)
            df = TechnicalIndicators.add_roc(df, periods=[10, 20])
            df = TechnicalIndicators.add_momentum(df, periods=[10, 20])

            # ボラティリティ系指標
            df = TechnicalIndicators.add_atr(df)
            df = TechnicalIndicators.add_volatility(df)

            # ボリューム系指標
            df = TechnicalIndicators.add_obv(df)
            df = TechnicalIndicators.add_volume_sma(df)

            # 価格変化
            df = TechnicalIndicators.add_returns(df)
            df = TechnicalIndicators.add_log_returns(df)

            # パターン・その他指標
            df = TechnicalIndicators.add_candlestick_patterns(df)
            df = TechnicalIndicators.add_pivot_points(df)
            df = TechnicalIndicators.add_price_channels(df)

            logger.info(f"Added {len(df.columns) - 6} technical indicators")  # 6はOHLCV+date

        except Exception as e:
            logger.error(f"Error adding indicators: {e}")

        return df

    # === トレンド系指標 ===

    @staticmethod
    def add_sma(df: pd.DataFrame, periods: list = [5, 10, 20, 50, 200]) -> pd.DataFrame:
        """単純移動平均（SMA）を追加"""
        for period in periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        return df

    @staticmethod
    def add_ema(df: pd.DataFrame, periods: list = [5, 10, 20, 50, 200]) -> pd.DataFrame:
        """指数移動平均（EMA）を追加"""
        for period in periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df

    @staticmethod
    def add_wma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """加重移動平均（WMA）を追加"""
        weights = np.arange(1, period + 1)
        df[f'wma_{period}'] = df['close'].rolling(period).apply(
            lambda prices: np.dot(prices, weights) / weights.sum(), raw=True
        )
        return df

    @staticmethod
    def add_macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """MACD（移動平均収束拡散）を追加"""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()

        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        return df

    @staticmethod
    def add_bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        std_dev: int = 2
    ) -> pd.DataFrame:
        """ボリンジャーバンドを追加"""
        df[f'bb_middle_{period}'] = df['close'].rolling(window=period).mean()
        rolling_std = df['close'].rolling(window=period).std()

        df[f'bb_upper_{period}'] = df[f'bb_middle_{period}'] + (rolling_std * std_dev)
        df[f'bb_lower_{period}'] = df[f'bb_middle_{period}'] - (rolling_std * std_dev)
        df[f'bb_width_{period}'] = df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']

        return df

    # === モメンタム系指標 ===

    @staticmethod
    def add_rsi(df: pd.DataFrame, periods: list = [14, 28]) -> pd.DataFrame:
        """RSI（相対力指数）を追加"""
        for period in periods:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        return df

    @staticmethod
    def add_stochastic(
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> pd.DataFrame:
        """ストキャスティクスを追加"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()

        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()

        return df

    @staticmethod
    def add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """CCI（商品チャネル指数）を追加"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())

        df[f'cci_{period}'] = (tp - sma_tp) / (0.015 * mad)

        return df

    @staticmethod
    def add_roc(df: pd.DataFrame, periods: list = [10, 20]) -> pd.DataFrame:
        """ROC（変化率）を追加"""
        for period in periods:
            df[f'roc_{period}'] = (
                (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100
            )
        return df

    @staticmethod
    def add_momentum(df: pd.DataFrame, periods: list = [10, 20]) -> pd.DataFrame:
        """モメンタムを追加"""
        for period in periods:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
        return df

    # === ボラティリティ系指標 ===

    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """ATR（平均真の範囲）を追加"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f'atr_{period}'] = true_range.rolling(window=period).mean()

        return df

    @staticmethod
    def add_volatility(df: pd.DataFrame, periods: list = [10, 20, 30]) -> pd.DataFrame:
        """ボラティリティ（標準偏差）を追加"""
        returns = df['close'].pct_change()

        for period in periods:
            df[f'volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(period)

        return df

    # === ボリューム系指標 ===

    @staticmethod
    def add_obv(df: pd.DataFrame) -> pd.DataFrame:
        """OBV（オンバランスボリューム）を追加"""
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i - 1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])

        df['obv'] = obv
        return df

    @staticmethod
    def add_volume_sma(df: pd.DataFrame, periods: list = [5, 10, 20]) -> pd.DataFrame:
        """出来高移動平均を追加"""
        for period in periods:
            df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
        return df

    # === 価格変化 ===

    @staticmethod
    def add_returns(df: pd.DataFrame, periods: list = [1, 5, 10]) -> pd.DataFrame:
        """リターンを追加"""
        for period in periods:
            df[f'return_{period}'] = df['close'].pct_change(periods=period)
        return df

    @staticmethod
    def add_log_returns(df: pd.DataFrame, periods: list = [1, 5, 10]) -> pd.DataFrame:
        """対数リターンを追加"""
        for period in periods:
            df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
        return df

    # === パターン認識 ===

    @staticmethod
    def add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """ローソク足パターンを追加"""
        # Doji（同時線）
        df['pattern_doji'] = (
            (np.abs(df['close'] - df['open']) / (df['high'] - df['low']) < 0.1)
        ).astype(int)

        # Hammer（ハンマー）
        body = np.abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)

        df['pattern_hammer'] = (
            (lower_shadow > 2 * body) & (upper_shadow < body)
        ).astype(int)

        # Shooting Star（流れ星）
        df['pattern_shooting_star'] = (
            (upper_shadow > 2 * body) & (lower_shadow < body)
        ).astype(int)

        return df

    # === その他の指標 ===

    @staticmethod
    def add_pivot_points(df: pd.DataFrame) -> pd.DataFrame:
        """ピボットポイントを追加"""
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['pivot_r1'] = 2 * df['pivot'] - df['low']
        df['pivot_r2'] = df['pivot'] + (df['high'] - df['low'])
        df['pivot_s1'] = 2 * df['pivot'] - df['high']
        df['pivot_s2'] = df['pivot'] - (df['high'] - df['low'])

        return df

    @staticmethod
    def add_price_channels(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """プライスチャネルを追加"""
        df[f'channel_high_{period}'] = df['high'].rolling(window=period).max()
        df[f'channel_low_{period}'] = df['low'].rolling(window=period).min()
        df[f'channel_middle_{period}'] = (
            df[f'channel_high_{period}'] + df[f'channel_low_{period}']
        ) / 2

        return df
