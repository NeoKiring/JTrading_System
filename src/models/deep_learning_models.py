"""
Deep Learning Models Module
ディープラーニングモデル（LSTM, GRU等）
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from .base_model import BaseModel
from ..utils.logger import get_logger
from ..utils.config import get_config

logger = get_logger(__name__)

# TensorFlow/Kerasのインポート（遅延インポート）
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Deep learning models will not be functional.")


class LSTMModel(BaseModel):
    """LSTMモデル"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初期化

        Args:
            config: モデル設定（省略時は設定ファイルから取得）
        """
        super().__init__()
        self.config = config or get_config('deep_learning.lstm', {})
        self.model = None
        self.history = None
        self.sequence_length = self.config.get('sequence_length', 60)
        self.feature_count = None

        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow is not installed. LSTM model cannot be used.")

    def _build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        LSTMモデルの構築

        Args:
            input_shape: 入力形状 (sequence_length, feature_count)

        Returns:
            keras.Model: 構築されたモデル
        """
        model = models.Sequential(name='LSTM_Model')

        # LSTM層の構成
        lstm_units = self.config.get('lstm_units', [128, 64, 32])
        dropout = self.config.get('dropout', 0.2)

        # 最初のLSTM層
        model.add(layers.LSTM(
            lstm_units[0],
            return_sequences=len(lstm_units) > 1,
            input_shape=input_shape,
            name='lstm_1'
        ))
        model.add(layers.Dropout(dropout, name='dropout_1'))

        # 追加のLSTM層
        for i, units in enumerate(lstm_units[1:], start=2):
            return_seq = i < len(lstm_units)
            model.add(layers.LSTM(
                units,
                return_sequences=return_seq,
                name=f'lstm_{i}'
            ))
            model.add(layers.Dropout(dropout, name=f'dropout_{i}'))

        # Dense層
        dense_units = self.config.get('dense_units', [32, 16])
        for i, units in enumerate(dense_units, start=1):
            model.add(layers.Dense(units, activation='relu', name=f'dense_{i}'))

        # 出力層
        model.add(layers.Dense(1, name='output'))

        return model

    def _prepare_sequences(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        シーケンスデータの準備

        Args:
            X: 特徴量データフレーム
            y: ターゲットシリーズ（予測時はNone）

        Returns:
            Tuple: (X_sequences, y_sequences)
        """
        X_array = X.values
        sequences = []
        targets = [] if y is not None else None

        for i in range(len(X_array) - self.sequence_length):
            sequences.append(X_array[i:i + self.sequence_length])
            if y is not None:
                targets.append(y.iloc[i + self.sequence_length])

        X_sequences = np.array(sequences)
        y_sequences = np.array(targets) if targets is not None else None

        return X_sequences, y_sequences

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        verbose: int = 1
    ):
        """
        モデルの訓練

        Args:
            X_train: 訓練用特徴量
            y_train: 訓練用ターゲット
            X_val: 検証用特徴量
            y_val: 検証用ターゲット
            verbose: 詳細度
        """
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow is not installed")

        logger.info("Preparing sequence data for LSTM training")

        # シーケンスデータの準備
        X_train_seq, y_train_seq = self._prepare_sequences(X_train, y_train)

        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val)
        else:
            X_val_seq, y_val_seq = None, None

        # 特徴量数の保存
        self.feature_count = X_train.shape[1]

        # モデルの構築
        input_shape = (self.sequence_length, self.feature_count)
        self.model = self._build_model(input_shape)

        # コンパイル
        learning_rate = self.config.get('learning_rate', 0.001)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )

        if verbose:
            self.model.summary()

        # コールバック設定
        callback_list = []

        # Early Stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss' if X_val_seq is not None else 'loss',
            patience=self.config.get('early_stopping_patience', 10),
            restore_best_weights=True,
            verbose=verbose
        )
        callback_list.append(early_stopping)

        # Learning Rate Reduction
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val_seq is not None else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=verbose
        )
        callback_list.append(reduce_lr)

        # 訓練
        logger.info("Training LSTM model...")

        epochs = self.config.get('epochs', 100)
        batch_size = self.config.get('batch_size', 32)

        validation_data = (X_val_seq, y_val_seq) if X_val_seq is not None else None

        self.history = self.model.fit(
            X_train_seq,
            y_train_seq,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=verbose
        )

        logger.info("LSTM model training completed")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測

        Args:
            X: 特徴量データフレーム

        Returns:
            np.ndarray: 予測値
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # シーケンスデータの準備
        X_seq, _ = self._prepare_sequences(X)

        # 予測
        predictions = self.model.predict(X_seq, verbose=0)

        return predictions.flatten()

    def save(self, filepath: str):
        """
        モデルの保存

        Args:
            filepath: 保存先パス
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        self.model.save(filepath)
        logger.info(f"LSTM model saved to {filepath}")

    def load(self, filepath: str):
        """
        モデルの読み込み

        Args:
            filepath: 読み込み元パス
        """
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow is not installed")

        self.model = keras.models.load_model(filepath)
        logger.info(f"LSTM model loaded from {filepath}")


class GRUModel(BaseModel):
    """GRUモデル"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初期化

        Args:
            config: モデル設定（省略時は設定ファイルから取得）
        """
        super().__init__()
        self.config = config or get_config('deep_learning.gru', {})
        self.model = None
        self.history = None
        self.sequence_length = self.config.get('sequence_length', 60)
        self.feature_count = None

        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow is not installed. GRU model cannot be used.")

    def _build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        GRUモデルの構築

        Args:
            input_shape: 入力形状 (sequence_length, feature_count)

        Returns:
            keras.Model: 構築されたモデル
        """
        model = models.Sequential(name='GRU_Model')

        # GRU層の構成
        gru_units = self.config.get('gru_units', [128, 64, 32])
        dropout = self.config.get('dropout', 0.2)

        # 最初のGRU層
        model.add(layers.GRU(
            gru_units[0],
            return_sequences=len(gru_units) > 1,
            input_shape=input_shape,
            name='gru_1'
        ))
        model.add(layers.Dropout(dropout, name='dropout_1'))

        # 追加のGRU層
        for i, units in enumerate(gru_units[1:], start=2):
            return_seq = i < len(gru_units)
            model.add(layers.GRU(
                units,
                return_sequences=return_seq,
                name=f'gru_{i}'
            ))
            model.add(layers.Dropout(dropout, name=f'dropout_{i}'))

        # Dense層
        dense_units = self.config.get('dense_units', [32, 16])
        for i, units in enumerate(dense_units, start=1):
            model.add(layers.Dense(units, activation='relu', name=f'dense_{i}'))

        # 出力層
        model.add(layers.Dense(1, name='output'))

        return model

    def _prepare_sequences(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        シーケンスデータの準備

        Args:
            X: 特徴量データフレーム
            y: ターゲットシリーズ（予測時はNone）

        Returns:
            Tuple: (X_sequences, y_sequences)
        """
        X_array = X.values
        sequences = []
        targets = [] if y is not None else None

        for i in range(len(X_array) - self.sequence_length):
            sequences.append(X_array[i:i + self.sequence_length])
            if y is not None:
                targets.append(y.iloc[i + self.sequence_length])

        X_sequences = np.array(sequences)
        y_sequences = np.array(targets) if targets is not None else None

        return X_sequences, y_sequences

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        verbose: int = 1
    ):
        """
        モデルの訓練

        Args:
            X_train: 訓練用特徴量
            y_train: 訓練用ターゲット
            X_val: 検証用特徴量
            y_val: 検証用ターゲット
            verbose: 詳細度
        """
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow is not installed")

        logger.info("Preparing sequence data for GRU training")

        # シーケンスデータの準備
        X_train_seq, y_train_seq = self._prepare_sequences(X_train, y_train)

        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val)
        else:
            X_val_seq, y_val_seq = None, None

        # 特徴量数の保存
        self.feature_count = X_train.shape[1]

        # モデルの構築
        input_shape = (self.sequence_length, self.feature_count)
        self.model = self._build_model(input_shape)

        # コンパイル
        learning_rate = self.config.get('learning_rate', 0.001)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )

        if verbose:
            self.model.summary()

        # コールバック設定
        callback_list = []

        # Early Stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss' if X_val_seq is not None else 'loss',
            patience=self.config.get('early_stopping_patience', 10),
            restore_best_weights=True,
            verbose=verbose
        )
        callback_list.append(early_stopping)

        # Learning Rate Reduction
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val_seq is not None else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=verbose
        )
        callback_list.append(reduce_lr)

        # 訓練
        logger.info("Training GRU model...")

        epochs = self.config.get('epochs', 100)
        batch_size = self.config.get('batch_size', 32)

        validation_data = (X_val_seq, y_val_seq) if X_val_seq is not None else None

        self.history = self.model.fit(
            X_train_seq,
            y_train_seq,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=verbose
        )

        logger.info("GRU model training completed")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測

        Args:
            X: 特徴量データフレーム

        Returns:
            np.ndarray: 予測値
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # シーケンスデータの準備
        X_seq, _ = self._prepare_sequences(X)

        # 予測
        predictions = self.model.predict(X_seq, verbose=0)

        return predictions.flatten()

    def save(self, filepath: str):
        """
        モデルの保存

        Args:
            filepath: 保存先パス
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        self.model.save(filepath)
        logger.info(f"GRU model saved to {filepath}")

    def load(self, filepath: str):
        """
        モデルの読み込み

        Args:
            filepath: 読み込み元パス
        """
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow is not installed")

        self.model = keras.models.load_model(filepath)
        logger.info(f"GRU model loaded from {filepath}")
