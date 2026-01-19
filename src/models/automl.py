"""
AutoML Module
自動機械学習モジュール（Optuna使用）
"""

import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path

from .ml_models import XGBoostModel, LightGBMModel, RandomForestModel
from ..utils.logger import get_logger
from ..utils.config import get_config

logger = get_logger(__name__)

# Optunaのログを制御
optuna.logging.set_verbosity(optuna.logging.WARNING)


class AutoML:
    """
    自動機械学習クラス

    Optunaを使用してハイパーパラメータを最適化し、
    最適なモデルを自動選択します。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初期化

        Args:
            config: AutoML設定（省略時は設定ファイルから取得）
        """
        self.config = config or get_config('automl', {})
        self.best_model = None
        self.best_params = None
        self.best_score = float('inf')
        self.study = None
        self.optimization_history = []

    def _get_metric_settings(self) -> Tuple[str, str]:
        """
        最適化メトリクスと方向を取得

        Returns:
            Tuple[str, str]: (metric_name, direction)
        """
        metric = self.config.get('metric', 'rmse').lower()
        if metric in {'rmse', 'mae'}:
            return metric, 'minimize'
        if metric == 'r2':
            return metric, 'maximize'
        logger.warning(f"Unknown metric '{metric}', defaulting to rmse")
        return 'rmse', 'minimize'

    @staticmethod
    def _calculate_metric(metric: str, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """指定されたメトリクスを計算"""
        if metric == 'mae':
            return mean_absolute_error(y_true, y_pred)
        if metric == 'r2':
            return r2_score(y_true, y_pred)
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def optimize_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """
        XGBoostのハイパーパラメータを最適化

        Args:
            X_train: 訓練用特徴量
            y_train: 訓練用ターゲット
            X_val: 検証用特徴量
            y_val: 検証用ターゲット
            n_trials: 試行回数

        Returns:
            Dict: 最適なパラメータ
        """
        metric_name, direction = self._get_metric_settings()
        logger.info(f"Optimizing XGBoost hyperparameters with metric={metric_name}...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
            }

            model = XGBoostModel(**params)
            model.train(X_train, y_train, verbose=False)

            y_pred = model.predict(X_val)
            return self._calculate_metric(metric_name, y_val, y_pred)

        study = optuna.create_study(
            direction=direction,
            study_name='xgboost_optimization'
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=self.config.get('optimization_timeout'),
            show_progress_bar=True
        )

        logger.info(f"Best {metric_name.upper()}: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        return study.best_params

    def optimize_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """
        LightGBMのハイパーパラメータを最適化

        Args:
            X_train: 訓練用特徴量
            y_train: 訓練用ターゲット
            X_val: 検証用特徴量
            y_val: 検証用ターゲット
            n_trials: 試行回数

        Returns:
            Dict: 最適なパラメータ
        """
        metric_name, direction = self._get_metric_settings()
        logger.info(f"Optimizing LightGBM hyperparameters with metric={metric_name}...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 200),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
            }

            model = LightGBMModel(**params)
            model.train(X_train, y_train, verbose=False)

            y_pred = model.predict(X_val)
            return self._calculate_metric(metric_name, y_val, y_pred)

        study = optuna.create_study(
            direction=direction,
            study_name='lightgbm_optimization'
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=self.config.get('optimization_timeout'),
            show_progress_bar=True
        )

        logger.info(f"Best {metric_name.upper()}: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        return study.best_params

    def optimize_randomforest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """
        RandomForestのハイパーパラメータを最適化

        Args:
            X_train: 訓練用特徴量
            y_train: 訓練用ターゲット
            X_val: 検証用特徴量
            y_val: 検証用ターゲット
            n_trials: 試行回数

        Returns:
            Dict: 最適なパラメータ
        """
        metric_name, direction = self._get_metric_settings()
        logger.info(f"Optimizing RandomForest hyperparameters with metric={metric_name}...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }

            model = RandomForestModel(**params)
            model.train(X_train, y_train, verbose=False)

            y_pred = model.predict(X_val)
            return self._calculate_metric(metric_name, y_val, y_pred)

        study = optuna.create_study(
            direction=direction,
            study_name='randomforest_optimization'
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=self.config.get('optimization_timeout'),
            show_progress_bar=True
        )

        logger.info(f"Best {metric_name.upper()}: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        return study.best_params

    def auto_select_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        models: Optional[List[str]] = None,
        n_trials_per_model: int = 30
    ) -> Tuple[Any, str, Dict[str, Any], Dict[str, float]]:
        """
        複数モデルを自動比較して最適なモデルを選択

        Args:
            X_train: 訓練用特徴量
            y_train: 訓練用ターゲット
            X_val: 検証用特徴量
            y_val: 検証用ターゲット
            models: 比較するモデルのリスト（省略時は全モデル）
            n_trials_per_model: モデルごとの試行回数

        Returns:
            Tuple: (最適モデル, モデル名, 最適パラメータ, 全モデルの結果)
        """
        if models is None:
            models = ['xgboost', 'lightgbm', 'randomforest']

        logger.info(f"Auto-selecting best model from: {models}")

        results = {}

        for model_name in models:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating {model_name.upper()}")
            logger.info(f"{'='*60}")

            try:
                if model_name == 'xgboost':
                    best_params = self.optimize_xgboost(
                        X_train, y_train, X_val, y_val,
                        n_trials=n_trials_per_model
                    )
                    model = XGBoostModel(**best_params)

                elif model_name == 'lightgbm':
                    best_params = self.optimize_lightgbm(
                        X_train, y_train, X_val, y_val,
                        n_trials=n_trials_per_model
                    )
                    model = LightGBMModel(**best_params)

                elif model_name == 'randomforest':
                    best_params = self.optimize_randomforest(
                        X_train, y_train, X_val, y_val,
                        n_trials=n_trials_per_model
                    )
                    model = RandomForestModel(**best_params)

                else:
                    logger.warning(f"Unknown model: {model_name}, skipping")
                    continue

                # 最適パラメータで訓練
                model.train(X_train, y_train, verbose=False)

                # 検証データで評価
                y_pred = model.predict(X_val)

                metrics = {
                    'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
                    'mae': mean_absolute_error(y_val, y_pred),
                    'r2': r2_score(y_val, y_pred)
                }

                results[model_name] = {
                    'model': model,
                    'params': best_params,
                    'metrics': metrics
                }

                logger.info(f"{model_name.upper()} Results:")
                logger.info(f"  RMSE: {metrics['rmse']:.4f}")
                logger.info(f"  MAE: {metrics['mae']:.4f}")
                logger.info(f"  R²: {metrics['r2']:.4f}")

            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                continue

        # 最適モデルの選択（RMSEが最小）
        best_model_name = min(results.keys(), key=lambda k: results[k]['metrics']['rmse'])
        best_result = results[best_model_name]

        logger.info(f"\n{'='*60}")
        logger.info(f"BEST MODEL: {best_model_name.upper()}")
        logger.info(f"  RMSE: {best_result['metrics']['rmse']:.4f}")
        logger.info(f"  MAE: {best_result['metrics']['mae']:.4f}")
        logger.info(f"  R²: {best_result['metrics']['r2']:.4f}")
        logger.info(f"{'='*60}\n")

        self.best_model = best_result['model']
        self.best_params = best_result['params']
        self.best_score = best_result['metrics']['rmse']

        # 全モデルの結果をまとめる
        all_results = {
            name: result['metrics']
            for name, result in results.items()
        }

        return (
            best_result['model'],
            best_model_name,
            best_result['params'],
            all_results
        )

    def save_best_model(self, filepath: str):
        """
        最適モデルを保存

        Args:
            filepath: 保存先パス
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet")

        # ディレクトリ作成
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # モデル保存
        self.best_model.save(filepath)

        # パラメータ保存
        params_path = filepath.replace('.joblib', '_params.joblib')
        joblib.dump(self.best_params, params_path)

        logger.info(f"Best model saved to {filepath}")
        logger.info(f"Best params saved to {params_path}")

    def load_best_model(self, filepath: str):
        """
        保存された最適モデルを読み込み

        Args:
            filepath: 読み込み元パス
        """
        # パラメータ読み込み
        params_path = filepath.replace('.joblib', '_params.joblib')

        if Path(params_path).exists():
            self.best_params = joblib.load(params_path)
            logger.info(f"Best params loaded from {params_path}")

        logger.info(f"Best model loaded from {filepath}")


class AutoMLPipeline:
    """AutoMLパイプライン（前処理含む）"""

    def __init__(self):
        """初期化"""
        self.automl = AutoML()
        self.preprocessor = None

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        prediction_days: int = 7,
        models: Optional[List[str]] = None,
        n_trials_per_model: int = 30
    ) -> Dict[str, Any]:
        """
        AutoMLパイプライン全体を実行

        Args:
            df: 株価データ
            symbol: 銘柄コード
            prediction_days: 予測日数
            models: 比較するモデルのリスト
            n_trials_per_model: モデルごとの試行回数

        Returns:
            Dict: パイプライン結果
        """
        from ..data.processors.data_preprocessor import DataPreprocessor

        logger.info(f"Starting AutoML pipeline for {symbol}")

        # データ前処理
        self.preprocessor = DataPreprocessor()

        X, y = self.preprocessor.prepare_training_data(
            df,
            prediction_days=prediction_days,
            add_technical_indicators=True,
            add_lag_features=True,
            add_rolling_features=True,
            scale_features=False
        )

        # 訓練/検証データ分割
        test_size = 1 - get_config('model.train_test_split', 0.8)
        X_train, X_val, y_train, y_val = self.preprocessor.train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        # スケーリング
        self.preprocessor.fit_scaler(X_train, method='standard')
        X_train = self.preprocessor.transform_features(X_train)
        X_val = self.preprocessor.transform_features(X_val)

        # AutoML実行
        best_model, best_name, best_params, all_results = self.automl.auto_select_model(
            X_train, y_train, X_val, y_val,
            models=models,
            n_trials_per_model=n_trials_per_model
        )

        # 結果をまとめる
        result = {
            'symbol': symbol,
            'best_model': best_model,
            'best_model_name': best_name,
            'best_params': best_params,
            'all_model_results': all_results,
            'preprocessor': self.preprocessor,
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val
        }

        logger.info("AutoML pipeline completed successfully")

        return result
