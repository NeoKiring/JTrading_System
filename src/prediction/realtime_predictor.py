"""
Real-Time Predictor Module
リアルタイム予測システム
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import json

from ..data.collectors.stock_collector import StockCollector
from ..data.processors.data_preprocessor import DataPreprocessor
from ..utils.logger import get_logger
from ..utils.config import get_config
from ..data.storage.database import DatabaseManager, Prediction

logger = get_logger(__name__)


class RealTimePredictor:
    """
    リアルタイム予測クラス

    最新データで予測を実行し、結果をキャッシュ・履歴管理します。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初期化

        Args:
            config: 予測設定（省略時は設定ファイルから取得）
        """
        self.config = config or get_config('realtime_prediction', {})
        self.collector = StockCollector()
        self.preprocessor = DataPreprocessor()
        self.db_manager = DatabaseManager()

        # キャッシュ
        self.prediction_cache = {}
        self.cache_ttl = self.config.get('cache_ttl_minutes', 30)

        # モデルキャッシュ
        self.loaded_models = {}

    def predict(
        self,
        symbol: str,
        model_path: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        指定銘柄の予測を実行

        Args:
            symbol: 銘柄コード
            model_path: モデルファイルパス（省略時は最新モデル）
            use_cache: キャッシュを使用するか

        Returns:
            Dict: 予測結果
        """
        # キャッシュチェック
        if use_cache:
            cached_result = self._get_cached_prediction(symbol)
            if cached_result is not None:
                logger.info(f"Using cached prediction for {symbol}")
                return cached_result

        logger.info(f"Generating new prediction for {symbol}")

        try:
            # 最新データ取得
            df = self.collector.collect(symbol)

            if df.empty or len(df) < 100:
                logger.warning(f"Insufficient data for {symbol}")
                return self._create_error_result(symbol, "Insufficient data")

            # モデル読み込み
            if model_path is None:
                model_path = self._find_latest_model(symbol)

            if not model_path or not Path(model_path).exists():
                logger.warning(f"No trained model found for {symbol}")
                return self._create_error_result(symbol, "No trained model")

            model = self._load_model(model_path)

            # 前処理と予測
            prediction_result = self._execute_prediction(df, symbol, model)

            # データベースに保存
            self._save_prediction_to_db(prediction_result)

            # キャッシュ更新
            self._update_cache(symbol, prediction_result)

            logger.info(f"Prediction completed for {symbol}: "
                       f"Current=¥{prediction_result['current_price']:.2f}, "
                       f"Predicted=¥{prediction_result['predicted_price']:.2f}, "
                       f"Change={prediction_result['predicted_change_pct']:.2f}%")

            return prediction_result

        except Exception as e:
            logger.error(f"Error predicting {symbol}: {e}")
            return self._create_error_result(symbol, str(e))

    def batch_predict(
        self,
        symbols: List[str],
        parallel: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        複数銘柄を一括予測

        Args:
            symbols: 銘柄コードリスト
            parallel: 並列実行するか

        Returns:
            Dict: {symbol: prediction_result}
        """
        logger.info(f"Batch prediction for {len(symbols)} symbols")

        results = {}

        if parallel:
            # TODO: 並列処理の実装（concurrent.futures使用）
            # 現在はシーケンシャル実行
            for symbol in symbols:
                results[symbol] = self.predict(symbol)
        else:
            for symbol in symbols:
                results[symbol] = self.predict(symbol)

        return results

    def _execute_prediction(
        self,
        df: pd.DataFrame,
        symbol: str,
        model: Any
    ) -> Dict[str, Any]:
        """
        予測を実行

        Args:
            df: 株価データ
            symbol: 銘柄コード
            model: 訓練済みモデル

        Returns:
            Dict: 予測結果
        """
        # 現在価格
        current_price = df['close'].iloc[-1]
        current_date = df['date'].iloc[-1] if 'date' in df.columns else datetime.now()

        # 特徴量準備
        prediction_days = get_config('model.prediction_days', 7)

        X, _ = self.preprocessor.prepare_training_data(
            df,
            prediction_days=prediction_days,
            add_technical_indicators=True,
            add_lag_features=True,
            add_rolling_features=True,
            scale_features=False
        )

        if X.empty:
            raise ValueError("Failed to prepare features")

        # スケーラー読み込み（モデルと一緒に保存されている想定）
        # スケーリング（訓練時と同じ方法）
        self.preprocessor.fit_scaler(X, method='standard')
        X_scaled = self.preprocessor.transform_features(X)

        # 最新データで予測
        X_latest = X_scaled.iloc[-1:].copy()
        predicted_price = model.predict(X_latest)[0]

        # 変化率計算
        predicted_change = predicted_price - current_price
        predicted_change_pct = (predicted_change / current_price) * 100

        # 目標日（予測日）
        target_date = current_date + timedelta(days=prediction_days)

        # 信頼区間の推定（簡易版：±標準偏差）
        # より正確な信頼区間には分位点回帰や予測分布が必要
        recent_volatility = df['close'].pct_change().std() * np.sqrt(prediction_days)
        confidence_range = predicted_price * recent_volatility

        result = {
            'symbol': symbol,
            'current_date': current_date,
            'current_price': float(current_price),
            'target_date': target_date,
            'predicted_price': float(predicted_price),
            'predicted_change': float(predicted_change),
            'predicted_change_pct': float(predicted_change_pct),
            'confidence_lower': float(predicted_price - confidence_range),
            'confidence_upper': float(predicted_price + confidence_range),
            'prediction_days': prediction_days,
            'model_type': model.__class__.__name__,
            'timestamp': datetime.now(),
            'status': 'success',
            'error': None
        }

        return result

    def _load_model(self, model_path: str) -> Any:
        """
        モデルを読み込み（キャッシュ対応）

        Args:
            model_path: モデルファイルパス

        Returns:
            訓練済みモデル
        """
        if model_path in self.loaded_models:
            return self.loaded_models[model_path]

        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        self.loaded_models[model_path] = model

        return model

    def _find_latest_model(self, symbol: str) -> Optional[str]:
        """
        最新のモデルファイルを検索

        Args:
            symbol: 銘柄コード

        Returns:
            モデルファイルパス（見つからない場合はNone）
        """
        models_dir = Path(get_config('paths.models', 'data/models'))

        # AutoMLモデルを優先
        automl_path = models_dir / 'ml' / f'{symbol}_automl_best.joblib'
        if automl_path.exists():
            return str(automl_path)

        # 通常のモデル
        ml_path = models_dir / 'ml' / f'{symbol}_xgboost.joblib'
        if ml_path.exists():
            return str(ml_path)

        return None

    def _get_cached_prediction(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        キャッシュから予測を取得

        Args:
            symbol: 銘柄コード

        Returns:
            キャッシュされた予測結果（有効期限切れの場合はNone）
        """
        if symbol not in self.prediction_cache:
            return None

        cached = self.prediction_cache[symbol]
        cache_time = cached.get('timestamp')

        if not cache_time:
            return None

        # TTLチェック
        elapsed = (datetime.now() - cache_time).total_seconds() / 60
        if elapsed > self.cache_ttl:
            del self.prediction_cache[symbol]
            return None

        return cached

    def _update_cache(self, symbol: str, result: Dict[str, Any]):
        """
        キャッシュを更新

        Args:
            symbol: 銘柄コード
            result: 予測結果
        """
        self.prediction_cache[symbol] = result

    def _save_prediction_to_db(self, result: Dict[str, Any]):
        """
        予測結果をデータベースに保存

        Args:
            result: 予測結果
        """
        if result['status'] != 'success':
            return

        try:
            with self.db_manager.get_session() as session:
                prediction = Prediction(
                    symbol=result['symbol'],
                    prediction_date=result['current_date'],
                    target_date=result['target_date'],
                    predicted_price=result['predicted_price'],
                    confidence_lower=result['confidence_lower'],
                    confidence_upper=result['confidence_upper'],
                    model_name=result['model_type']
                )
                session.add(prediction)

        except Exception as e:
            logger.error(f"Error saving prediction to database: {e}")

    def _create_error_result(self, symbol: str, error: str) -> Dict[str, Any]:
        """
        エラー結果を作成

        Args:
            symbol: 銘柄コード
            error: エラーメッセージ

        Returns:
            エラー結果
        """
        return {
            'symbol': symbol,
            'current_date': datetime.now(),
            'current_price': None,
            'target_date': None,
            'predicted_price': None,
            'predicted_change': None,
            'predicted_change_pct': None,
            'confidence_lower': None,
            'confidence_upper': None,
            'prediction_days': None,
            'model_type': None,
            'timestamp': datetime.now(),
            'status': 'error',
            'error': error
        }

    def get_prediction_history(
        self,
        symbol: str,
        days: int = 30
    ) -> pd.DataFrame:
        """
        予測履歴を取得

        Args:
            symbol: 銘柄コード
            days: 過去何日分を取得するか

        Returns:
            予測履歴DataFrame
        """
        try:
            with self.db_manager.get_session() as session:
                start_date = datetime.now() - timedelta(days=days)

                predictions = session.query(Prediction).filter(
                    Prediction.symbol == symbol,
                    Prediction.prediction_date >= start_date
                ).order_by(Prediction.prediction_date.desc()).all()

                if not predictions:
                    return pd.DataFrame()

                data = [{
                    'prediction_date': p.prediction_date,
                    'target_date': p.target_date,
                    'predicted_price': p.predicted_price,
                    'confidence_lower': p.confidence_lower,
                    'confidence_upper': p.confidence_upper,
                    'model_name': p.model_name
                } for p in predictions]

                return pd.DataFrame(data)

        except Exception as e:
            logger.error(f"Error retrieving prediction history: {e}")
            return pd.DataFrame()

    def evaluate_prediction_accuracy(
        self,
        symbol: str,
        days: int = 30
    ) -> Dict[str, float]:
        """
        予測精度を評価

        過去の予測と実際の価格を比較します。

        Args:
            symbol: 銘柄コード
            days: 評価期間（日数）

        Returns:
            Dict: 精度メトリクス
        """
        # 予測履歴取得
        history_df = self.get_prediction_history(symbol, days=days)

        if history_df.empty:
            return {}

        # 実際の株価取得
        actual_df = self.collector.collect(symbol)

        if actual_df.empty:
            return {}

        # 予測と実際の価格を照合
        matched_predictions = []
        matched_actuals = []

        for _, pred_row in history_df.iterrows():
            target_date = pred_row['target_date']
            predicted_price = pred_row['predicted_price']

            # 実際の価格を検索（±1日の範囲）
            actual_rows = actual_df[
                (actual_df['date'] >= target_date - timedelta(days=1)) &
                (actual_df['date'] <= target_date + timedelta(days=1))
            ]

            if not actual_rows.empty:
                actual_price = actual_rows.iloc[0]['close']
                matched_predictions.append(predicted_price)
                matched_actuals.append(actual_price)

        if not matched_predictions:
            return {}

        # メトリクス計算
        predictions_array = np.array(matched_predictions)
        actuals_array = np.array(matched_actuals)

        mae = np.mean(np.abs(predictions_array - actuals_array))
        rmse = np.sqrt(np.mean((predictions_array - actuals_array) ** 2))
        mape = np.mean(np.abs((actuals_array - predictions_array) / actuals_array)) * 100

        # 方向性の精度（上がる/下がるの予測精度）
        direction_correct = 0
        for i in range(len(matched_predictions) - 1):
            pred_direction = matched_predictions[i+1] > matched_predictions[i]
            actual_direction = matched_actuals[i+1] > matched_actuals[i]
            if pred_direction == actual_direction:
                direction_correct += 1

        direction_accuracy = (direction_correct / max(len(matched_predictions) - 1, 1)) * 100

        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'direction_accuracy': float(direction_accuracy),
            'num_samples': len(matched_predictions)
        }

        logger.info(f"Prediction accuracy for {symbol}: "
                   f"MAE={mae:.2f}, RMSE={rmse:.2f}, "
                   f"MAPE={mape:.2f}%, Direction={direction_accuracy:.1f}%")

        return metrics
