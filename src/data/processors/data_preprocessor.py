"""
Data Preprocessor Module
データ前処理と特徴量エンジニアリング
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Optional, Tuple
from .technical_indicators import TechnicalIndicators
from ...utils.logger import get_logger
from ...utils.config import get_config
from ...utils.helpers import create_lag_features, create_rolling_features

logger = get_logger(__name__)


class DataPreprocessor:
    """データ前処理クラス"""

    def __init__(self):
        """初期化"""
        self.scaler = None
        self.feature_columns = None
        self.target_column = 'target'

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        prediction_days: int = 7,
        add_technical_indicators: bool = True,
        add_lag_features: bool = True,
        add_rolling_features: bool = True,
        scale_features: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        学習用データを準備

        Args:
            df: 株価データ
            prediction_days: 予測日数
            add_technical_indicators: テクニカル指標を追加するか
            add_lag_features: ラグ特徴量を追加するか
            add_rolling_features: ローリング特徴量を追加するか
            scale_features: 特徴量をスケーリングするか

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (特徴量, ターゲット)
        """
        logger.info("Preparing training data...")

        df = df.copy()

        # ターゲット変数の作成（N日後の終値）
        df[self.target_column] = df['close'].shift(-prediction_days)

        # テクニカル指標の追加
        if add_technical_indicators:
            logger.info("Adding technical indicators...")
            df = TechnicalIndicators.add_all_indicators(df)

        # ラグ特徴量の追加
        if add_lag_features:
            lag_periods = get_config('model.features.lag_periods', [1, 2, 3, 5, 7, 14])
            logger.info(f"Adding lag features: {lag_periods}")
            df = create_lag_features(
                df,
                columns=['close', 'volume', 'high', 'low'],
                lags=lag_periods
            )

        # ローリング特徴量の追加
        if add_rolling_features:
            logger.info("Adding rolling features...")
            df = create_rolling_features(
                df,
                columns=['close', 'volume'],
                windows=[5, 10, 20],
                functions=['mean', 'std', 'min', 'max']
            )

        # 欠損値の処理
        df = self._handle_missing_values(df)

        # 無限大の処理
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()

        # 特徴量とターゲットの分離
        feature_cols = [col for col in df.columns
                       if col not in ['symbol', 'date', self.target_column]]

        X = df[feature_cols]
        y = df[self.target_column]

        # 特徴量カラムを保存
        self.feature_columns = feature_cols

        # スケーリング
        if scale_features:
            X = self._scale_features(X)

        logger.info(f"Prepared data: {len(X)} samples, {len(feature_cols)} features")

        return X, y

    def prepare_prediction_data(
        self,
        df: pd.DataFrame,
        add_technical_indicators: bool = True,
        add_lag_features: bool = True,
        add_rolling_features: bool = True,
        scale_features: bool = True
    ) -> pd.DataFrame:
        """
        予測用データを準備

        Args:
            df: 株価データ
            add_technical_indicators: テクニカル指標を追加するか
            add_lag_features: ラグ特徴量を追加するか
            add_rolling_features: ローリング特徴量を追加するか
            scale_features: 特徴量をスケーリングするか

        Returns:
            pd.DataFrame: 特徴量
        """
        df = df.copy()

        # テクニカル指標の追加
        if add_technical_indicators:
            df = TechnicalIndicators.add_all_indicators(df)

        # ラグ特徴量の追加
        if add_lag_features:
            lag_periods = get_config('model.features.lag_periods', [1, 2, 3, 5, 7, 14])
            df = create_lag_features(
                df,
                columns=['close', 'volume', 'high', 'low'],
                lags=lag_periods
            )

        # ローリング特徴量の追加
        if add_rolling_features:
            df = create_rolling_features(
                df,
                columns=['close', 'volume'],
                windows=[5, 10, 20],
                functions=['mean', 'std', 'min', 'max']
            )

        # 欠損値の処理
        df = self._handle_missing_values(df)

        # 無限大の処理
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()

        # 特徴量の選択
        if self.feature_columns:
            # 学習時と同じ特徴量を使用
            missing_cols = set(self.feature_columns) - set(df.columns)
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
                # 不足カラムを0で埋める
                for col in missing_cols:
                    df[col] = 0

            X = df[self.feature_columns]
        else:
            # 初回の場合は全特徴量を使用
            feature_cols = [col for col in df.columns
                           if col not in ['symbol', 'date', self.target_column]]
            X = df[feature_cols]

        # スケーリング
        if scale_features and self.scaler is not None:
            X = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )

        return X

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        欠損値を処理

        Args:
            df: データフレーム

        Returns:
            pd.DataFrame: 処理済みデータ
        """
        # 前方埋め（Forward Fill）
        df = df.fillna(method='ffill', limit=5)

        # 後方埋め（Backward Fill）
        df = df.fillna(method='bfill', limit=5)

        # それでも残る欠損値は削除
        initial_count = len(df)
        df = df.dropna()
        removed_count = initial_count - len(df)

        if removed_count > 0:
            logger.warning(f"Removed {removed_count} rows with missing values")

        return df

    def _scale_features(self, X: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        特徴量をスケーリング

        Args:
            X: 特徴量
            method: スケーリング方法（'standard', 'minmax'）

        Returns:
            pd.DataFrame: スケーリング済み特徴量
        """
        if method == 'standard':
            if self.scaler is None:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)

        elif method == 'minmax':
            if self.scaler is None:
                self.scaler = MinMaxScaler()
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)

        else:
            raise ValueError(f"Unknown scaling method: {method}")

        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    def train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        shuffle: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        学習データとテストデータに分割

        Args:
            X: 特徴量
            y: ターゲット
            test_size: テストデータの割合
            shuffle: シャッフルするか（時系列データではFalse推奨）

        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        if shuffle:
            # シャッフルする場合
            from sklearn.model_selection import train_test_split
            return train_test_split(X, y, test_size=test_size, random_state=42)
        else:
            # 時系列データの場合は順序を保つ
            split_idx = int(len(X) * (1 - test_size))

            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]

            logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

            return X_train, X_test, y_train, y_test
