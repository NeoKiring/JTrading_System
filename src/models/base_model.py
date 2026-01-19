"""
Base Model Module
モデルの基底クラス
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
import joblib
from pathlib import Path
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """モデルの基底クラス"""

    def __init__(self, name: str = "BaseModel"):
        """
        初期化

        Args:
            name: モデル名
        """
        self.name = name
        self.model = None
        self.is_trained = False
        self.logger = get_logger(name)

    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        モデルを訓練

        Args:
            X_train: 学習用特徴量
            y_train: 学習用ターゲット
            X_val: 検証用特徴量（オプション）
            y_val: 検証用ターゲット（オプション）
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        予測を実行

        Args:
            X: 特徴量

        Returns:
            予測結果
        """
        pass

    def save(self, filepath: str):
        """
        モデルを保存

        Args:
            filepath: 保存先パス
        """
        if not self.is_trained:
            self.logger.warning("Model is not trained yet")
            return

        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, filepath)
            self.logger.info(f"Model saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")

    def load(self, filepath: str):
        """
        モデルを読み込み

        Args:
            filepath: モデルファイルパス
        """
        try:
            self.model = joblib.load(filepath)
            self.is_trained = True
            self.logger.info(f"Model loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")

    def get_params(self) -> Dict[str, Any]:
        """
        モデルパラメータを取得

        Returns:
            Dict: パラメータ辞書
        """
        if self.model and hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return {}

    def set_params(self, **params):
        """
        モデルパラメータを設定

        Args:
            **params: パラメータ
        """
        if self.model and hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
