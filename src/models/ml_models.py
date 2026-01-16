"""
Machine Learning Models Module
機械学習モデル（XGBoost, LightGBM, Random Forest）
"""

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from typing import Optional, Dict, Any
from .base_model import BaseModel
from ..utils.config import get_model_config


class XGBoostModel(BaseModel):
    """XGBoostモデル"""

    def __init__(self, **kwargs):
        """
        初期化

        Args:
            **kwargs: XGBoostパラメータ（設定ファイルの値を上書き）
        """
        super().__init__(name="XGBoostModel")

        # 設定ファイルからパラメータを読み込み
        params = get_model_config('xgboost')
        params.update(kwargs)

        self.model = xgb.XGBRegressor(**params)

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        early_stopping_rounds: int = 10,
        verbose: bool = True
    ):
        """
        モデルを訓練

        Args:
            X_train: 学習用特徴量
            y_train: 学習用ターゲット
            X_val: 検証用特徴量
            y_val: 検証用ターゲット
            early_stopping_rounds: 早期停止ラウンド数
            verbose: 詳細出力
        """
        self.logger.info("Training XGBoost model...")

        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]

            # XGBoost 2.0以降の早期停止の実装
            callbacks = [xgb.callback.EarlyStopping(rounds=early_stopping_rounds)]

            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                callbacks=callbacks,
                verbose=verbose
            )
        else:
            self.model.fit(X_train, y_train, verbose=verbose)

        self.is_trained = True
        self.logger.info("XGBoost model training completed")

    def predict(self, X):
        """
        予測を実行

        Args:
            X: 特徴量

        Returns:
            np.ndarray: 予測結果
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet")

        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        特徴量重要度を取得

        Returns:
            Dict: 特徴量重要度
        """
        if not self.is_trained:
            return {}

        importance = self.model.feature_importances_
        feature_names = self.model.feature_names_in_ if hasattr(self.model, 'feature_names_in_') else None

        if feature_names is not None:
            return dict(zip(feature_names, importance))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(importance)}


class LightGBMModel(BaseModel):
    """LightGBMモデル"""

    def __init__(self, **kwargs):
        """
        初期化

        Args:
            **kwargs: LightGBMパラメータ
        """
        super().__init__(name="LightGBMModel")

        params = get_model_config('lightgbm')
        params.update(kwargs)

        self.model = lgb.LGBMRegressor(**params)

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        early_stopping_rounds: int = 10,
        verbose: bool = True
    ):
        """モデルを訓練"""
        self.logger.info("Training LightGBM model...")

        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]

            # コールバックの設定
            callbacks = [lgb.early_stopping(early_stopping_rounds)]
            if verbose:
                callbacks.append(lgb.log_evaluation(period=10))

            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                callbacks=callbacks
            )
        else:
            self.model.fit(X_train, y_train)

        self.is_trained = True
        self.logger.info("LightGBM model training completed")

    def predict(self, X):
        """予測を実行"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet")

        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """特徴量重要度を取得"""
        if not self.is_trained:
            return {}

        importance = self.model.feature_importances_
        feature_names = self.model.feature_name_ if hasattr(self.model, 'feature_name_') else None

        if feature_names is not None:
            return dict(zip(feature_names, importance))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(importance)}


class RandomForestModel(BaseModel):
    """Random Forestモデル"""

    def __init__(self, **kwargs):
        """
        初期化

        Args:
            **kwargs: Random Forestパラメータ
        """
        super().__init__(name="RandomForestModel")

        params = get_model_config('random_forest')
        params.update(kwargs)

        self.model = RandomForestRegressor(**params)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """モデルを訓練"""
        self.logger.info("Training Random Forest model...")

        self.model.fit(X_train, y_train)

        self.is_trained = True
        self.logger.info("Random Forest model training completed")

    def predict(self, X):
        """予測を実行"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet")

        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """特徴量重要度を取得"""
        if not self.is_trained:
            return {}

        importance = self.model.feature_importances_
        feature_names = self.model.feature_names_in_ if hasattr(self.model, 'feature_names_in_') else None

        if feature_names is not None:
            return dict(zip(feature_names, importance))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(importance)}


def evaluate_model(model: BaseModel, X_test, y_test) -> Dict[str, float]:
    """
    モデルを評価

    Args:
        model: モデルインスタンス
        X_test: テスト用特徴量
        y_test: テスト用ターゲット

    Returns:
        Dict: 評価メトリクス
    """
    y_pred = model.predict(X_test)

    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }

    # MAPE（Mean Absolute Percentage Error）
    # ゼロ除算を防ぐため、epsilon 以上の値のみで計算
    epsilon = 1e-10
    y_test_array = np.array(y_test)
    y_pred_array = np.array(y_pred)

    # 絶対値がepsilon以上の要素のみを使用
    mask = np.abs(y_test_array) >= epsilon
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_test_array[mask] - y_pred_array[mask]) / y_test_array[mask])) * 100
    else:
        # すべての値がepsilon未満の場合はNaNを設定
        mape = np.nan

    metrics['mape'] = mape

    # sMAPE（Symmetric Mean Absolute Percentage Error）
    # より安定した代替指標
    denominator = (np.abs(y_test_array) + np.abs(y_pred_array)) / 2
    mask_smape = denominator >= epsilon
    if np.sum(mask_smape) > 0:
        smape = np.mean(np.abs(y_test_array[mask_smape] - y_pred_array[mask_smape]) / denominator[mask_smape]) * 100
    else:
        smape = np.nan

    metrics['smape'] = smape

    return metrics
