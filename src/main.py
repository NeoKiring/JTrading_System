"""
Main Program
JTrading System メインプログラム
"""

import argparse
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger, get_logger
from src.utils.config import get_config, get_symbols
from src.data.collectors.stock_collector import StockCollector
from src.data.collectors.news_collector import NewsCollector
from src.data.processors.data_preprocessor import DataPreprocessor
from src.data.processors.sentiment_analyzer import SentimentAnalyzer
from src.models.ml_models import XGBoostModel, evaluate_model
from src.models.automl import AutoMLPipeline
from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.report_generator import ReportGenerator
from src.prediction.realtime_predictor import RealTimePredictor
from src.prediction.prediction_scheduler import PredictionScheduler, SmartScheduler
from src.prediction.alert_manager import AlertManager
from src.gui.main_window import launch_gui
from src.gui.main_window_enhanced import launch_enhanced_gui


def setup_system():
    """システムのセットアップ"""
    # ロガーのセットアップ
    setup_logger(
        log_level=get_config('logging.level', 'INFO'),
        log_format=get_config('logging.format'),
        rotation=get_config('logging.rotation'),
        retention=get_config('logging.retention'),
        app_log_path=get_config('logging.files.app'),
        error_log_path=get_config('logging.files.error'),
        trading_log_path=get_config('logging.files.trading')
    )

    logger = get_logger(__name__)
    logger.info("=" * 60)
    logger.info(f"{get_config('app.name')} v{get_config('app.version')}")
    logger.info("=" * 60)

    return logger


def collect_data_workflow(symbols: list, logger):
    """データ収集ワークフロー"""
    logger.info("Starting data collection workflow...")

    collector = StockCollector()
    results = collector.collect_multiple(
        symbols=[s['symbol'] for s in symbols],
        save_to_db=True
    )

    logger.info(f"Data collection completed for {len(results)} symbols")
    return results


def train_model_workflow(symbol: str, logger):
    """モデル訓練ワークフロー"""
    logger.info(f"Starting model training workflow for {symbol}...")

    # データ収集
    collector = StockCollector()
    df = collector.collect(symbol, save_to_db=False)

    if df is None or df.empty:
        logger.error(f"No data available for {symbol}")
        return None

    # データ前処理
    preprocessor = DataPreprocessor()
    prediction_days = get_config('model.prediction_days', 7)

    X, y = preprocessor.prepare_training_data(
        df,
        prediction_days=prediction_days,
        add_technical_indicators=True,
        add_lag_features=True,
        add_rolling_features=True,
        scale_features=False  # スケーリングは分割後に実施（データリーケージ防止）
    )

    # 学習/テストデータ分割
    test_size = 1 - get_config('model.train_test_split', 0.8)
    X_train, X_test, y_train, y_test = preprocessor.train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    # スケーリング（訓練データでfitし、テストデータにtransform）
    preprocessor.fit_scaler(X_train, method='standard')
    X_train = preprocessor.transform_features(X_train)
    X_test = preprocessor.transform_features(X_test)

    # モデル訓練
    model = XGBoostModel()
    model.train(X_train, y_train, X_test, y_test)

    # モデル評価
    metrics = evaluate_model(model, X_test, y_test)

    logger.info("Model Training Results:")
    logger.info(f"  RMSE: {metrics['rmse']:.2f}")
    logger.info(f"  MAE: {metrics['mae']:.2f}")
    logger.info(f"  R²: {metrics['r2']:.4f}")

    # MAPEとsMAPEの表示（NaNの場合は"N/A"と表示）
    import math
    mape_str = f"{metrics['mape']:.2f}%" if not math.isnan(metrics['mape']) else "N/A (zero values in y_test)"
    smape_str = f"{metrics['smape']:.2f}%" if not math.isnan(metrics['smape']) else "N/A"

    logger.info(f"  MAPE: {mape_str}")
    logger.info(f"  sMAPE: {smape_str}")

    # モデル保存
    model_path = f"data/models/ml/{symbol}_xgboost.joblib"
    model.save(model_path)

    return model, preprocessor, metrics


def backtest_workflow(symbol: str, logger):
    """バックテストワークフロー"""
    logger.info(f"Starting backtest workflow for {symbol}...")

    # データ収集
    collector = StockCollector()
    df = collector.collect(symbol, save_to_db=False)

    if df is None or df.empty:
        logger.error(f"No data available for {symbol}")
        return None

    # データ前処理
    preprocessor = DataPreprocessor()
    prediction_days = get_config('model.prediction_days', 7)

    X, y = preprocessor.prepare_training_data(
        df,
        prediction_days=prediction_days,
        add_technical_indicators=True,
        add_lag_features=True,
        add_rolling_features=True,
        scale_features=False  # スケーリングは分割後に実施（データリーケージ防止）
    )

    # モデル訓練（簡易版）
    test_size = 1 - get_config('model.train_test_split', 0.8)
    X_train, X_test, y_train, y_test = preprocessor.train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    # スケーリング（訓練データでfitし、テストデータにtransform）
    preprocessor.fit_scaler(X_train, method='standard')
    X_train = preprocessor.transform_features(X_train)
    X_test = preprocessor.transform_features(X_test)

    model = XGBoostModel()
    model.train(X_train, y_train, verbose=False)

    # テストデータで予測
    y_pred = model.predict(X_test)

    # バックテスト実行
    test_df = df.iloc[-len(y_test):].copy()
    test_df = test_df.reset_index(drop=True)

    engine = BacktestEngine()
    results = engine.run(test_df, y_pred, threshold=0.02)

    logger.info("Backtest Results:")
    logger.info(f"  Initial Value: ¥{results['initial_value']:,.0f}")
    logger.info(f"  Final Value: ¥{results['final_value']:,.0f}")
    logger.info(f"  Total Return: {results['total_return_pct']:.2f}%")
    logger.info(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    logger.info(f"  Win Rate: {results['win_rate_pct']:.2f}%")
    logger.info(f"  Number of Trades: {results['num_trades']}")

    report_format = get_config('reports.default_format', 'excel')
    report_generator = ReportGenerator()
    if report_format == 'pdf':
        report_path = report_generator.generate_backtest_report_pdf(results, symbol)
    else:
        report_path = report_generator.generate_backtest_report_excel(results, symbol)
    if report_path:
        logger.info(f"Backtest report generated: {report_path}")

    return results


def news_workflow(symbols: list, logger):
    """ニュース収集・感情分析ワークフロー"""
    logger.info("Starting news collection and sentiment analysis workflow...")

    # symbols が dict のリストの場合、symbol 文字列のリストに変換
    if symbols and isinstance(symbols[0], dict):
        symbol_list = [s['symbol'] for s in symbols]
    else:
        symbol_list = symbols

    # ニュース収集
    news_collector = NewsCollector()
    logger.info(f"Collecting news for {len(symbol_list)} symbols")

    news_collector.update_news_batch(symbol_list, days=get_config('news.default_days', 7))

    # 感情分析
    sentiment_analyzer = SentimentAnalyzer()
    logger.info("Analyzing sentiment for all collected news...")

    for symbol in symbol_list:
        try:
            sentiment_analyzer.update_database_sentiment(symbol=symbol)
            summary = sentiment_analyzer.get_sentiment_summary(symbol=symbol)

            logger.info(f"\n{symbol} Sentiment Summary:")
            logger.info(f"  Total Articles: {summary.get('total_articles', 0)}")
            logger.info(f"  Average Sentiment: {summary.get('average_sentiment', 0):.3f}")
            logger.info(f"  Positive: {summary.get('positive_count', 0)} ({summary.get('positive_ratio', 0)*100:.1f}%)")
            logger.info(f"  Neutral: {summary.get('neutral_count', 0)}")
            logger.info(f"  Negative: {summary.get('negative_count', 0)} ({summary.get('negative_ratio', 0)*100:.1f}%)")

        except Exception as e:
            logger.error(f"Error processing sentiment for {symbol}: {e}")

    logger.info("\nNews workflow completed successfully!")


def automl_workflow(symbol: str, logger):
    """AutoMLワークフロー（ハイパーパラメータ最適化＋モデル比較）"""
    logger.info("=" * 60)
    logger.info(f"Starting AutoML workflow for {symbol}")
    logger.info("=" * 60)

    # データ収集
    logger.info("\n[1/3] Data Collection")
    collector = StockCollector()
    df = collector.collect(symbol)

    if df is None or df.empty:
        logger.error(f"Failed to collect data for {symbol}")
        return

    # AutoMLパイプライン実行
    logger.info("\n[2/3] AutoML Optimization")
    pipeline = AutoMLPipeline()

    models = get_config('automl.default_models', ['xgboost', 'lightgbm', 'randomforest'])
    n_trials = get_config('automl.n_trials_per_model', 30)

    result = pipeline.run(
        df=df,
        symbol=symbol,
        models=models,
        n_trials_per_model=n_trials
    )

    # 結果表示
    logger.info("\n[3/3] Results")
    logger.info(f"Best Model: {result['best_model_name'].upper()}")
    logger.info("\nModel Comparison:")

    for model_name, metrics in result['all_model_results'].items():
        logger.info(f"\n{model_name.upper()}:")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  MAE: {metrics['mae']:.4f}")
        logger.info(f"  R²: {metrics['r2']:.4f}")

    # 最適モデルの保存
    logger.info("\nSaving best model...")
    model_path = f"data/models/ml/{symbol}_automl_best.joblib"
    pipeline.automl.save_best_model(model_path)

    logger.info("\n" + "=" * 60)
    logger.info("AutoML workflow completed successfully!")
    logger.info("=" * 60)

    return result


def full_workflow(symbol: str, logger):
    """フルワークフロー（データ収集→訓練→バックテスト）"""
    logger.info("=" * 60)
    logger.info(f"Starting full workflow for {symbol}")
    logger.info("=" * 60)

    # データ収集
    logger.info("\n[1/3] Data Collection")
    collector = StockCollector()
    df = collector.collect(symbol, save_to_db=True)

    if df is None or df.empty:
        logger.error(f"Failed to collect data for {symbol}")
        return

    # モデル訓練
    logger.info("\n[2/3] Model Training")
    model, preprocessor, metrics = train_model_workflow(symbol, logger)

    if model is None:
        logger.error("Model training failed")
        return

    # バックテスト
    logger.info("\n[3/3] Backtesting")
    results = backtest_workflow(symbol, logger)

    logger.info("\n" + "=" * 60)
    logger.info("Full workflow completed successfully!")
    logger.info("=" * 60)


def realtime_prediction_workflow(symbols: list, logger, mode: str = 'once'):
    """リアルタイム予測ワークフロー

    Args:
        symbols: 予測する銘柄リスト
        logger: ロガー
        mode: 実行モード ('once': 1回のみ, 'schedule': スケジュール実行, 'smart': スマートスケジュール)
    """
    logger.info("=" * 60)
    logger.info("Starting Real-Time Prediction Workflow")
    logger.info("=" * 60)

    # 銘柄リストの処理
    if symbols and isinstance(symbols[0], dict):
        symbol_list = [s['symbol'] for s in symbols]
    else:
        symbol_list = symbols

    # アラートマネージャー初期化
    alert_manager = AlertManager()

    # アラートコールバック（コンソール出力）
    def on_alert(alert):
        """アラート通知コールバック"""
        logger.warning(f"⚠️ ALERT: {alert}")

    alert_manager.register_callback(on_alert)

    if mode == 'once':
        # 1回のみ実行
        logger.info(f"\n[Mode: One-time] Predicting {len(symbol_list)} symbols\n")

        predictor = RealTimePredictor()

        for symbol in symbol_list:
            logger.info(f"\n{'='*40}")
            logger.info(f"Predicting: {symbol}")
            logger.info(f"{'='*40}")

            # 予測実行
            result = predictor.predict(symbol)

            if result['status'] == 'success':
                logger.info(f"✓ Prediction successful")
                logger.info(f"  Current Price: ¥{result['current_price']:,.2f}")
                logger.info(f"  Predicted Price ({result['prediction_days']} days): ¥{result['predicted_price']:,.2f}")
                logger.info(f"  Expected Change: {result['predicted_change_pct']:+.2f}%")
                logger.info(f"  Confidence Range: ¥{result['confidence_lower']:,.2f} - ¥{result['confidence_upper']:,.2f}")
                logger.info(f"  Model: {result['model_type']}")

                # アラートチェック
                alerts = alert_manager.check_prediction(result)
                if alerts:
                    logger.info(f"  Generated {len(alerts)} alert(s)")

            else:
                logger.error(f"✗ Prediction failed: {result['error']}")

        # アラートサマリー
        summary = alert_manager.get_summary()
        logger.info(f"\n{'='*60}")
        logger.info("Alert Summary")
        logger.info(f"{'='*60}")
        logger.info(f"  Total Alerts: {summary['total']}")
        logger.info(f"  By Level: {summary['by_level']}")
        logger.info(f"  By Type: {summary['by_type']}")

    elif mode == 'schedule':
        # 定期実行
        logger.info(f"\n[Mode: Scheduled] Starting prediction scheduler\n")

        scheduler = PredictionScheduler()

        # 予測完了時のコールバック
        def on_prediction_complete(symbol, result):
            if result['status'] == 'success':
                logger.info(f"✓ {symbol}: {result['predicted_change_pct']:+.2f}%")

                # アラートチェック
                alert_manager.check_prediction(result)

        # バッチ完了時のコールバック
        def on_batch_complete(results):
            success_count = sum(1 for r in results.values() if r['status'] == 'success')
            logger.info(f"\n✓ Batch prediction completed: {success_count}/{len(results)} succeeded")

            # アラートサマリー
            summary = alert_manager.get_summary()
            if summary['recent_count'] > 0:
                logger.warning(f"⚠️ {summary['recent_count']} new alert(s)")

        scheduler.set_callbacks(
            on_prediction_complete=on_prediction_complete,
            on_batch_complete=on_batch_complete
        )

        # スケジューラー開始
        interval_minutes = get_config('realtime_prediction.interval_minutes', 60)
        scheduler.start(symbols=symbol_list, interval_minutes=interval_minutes)

        logger.info(f"Scheduler started (interval: {interval_minutes} minutes)")
        logger.info("Press Ctrl+C to stop...\n")

        try:
            # メインスレッドを維持
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\nStopping scheduler...")
            scheduler.stop()
            logger.info("Scheduler stopped")

    elif mode == 'smart':
        # スマートスケジュール実行
        logger.info(f"\n[Mode: Smart Schedule] Starting smart prediction scheduler\n")
        logger.info("Scheduled times: 09:15, 11:15, 12:45, 15:15\n")

        scheduler = SmartScheduler()

        # 予測完了時のコールバック
        def on_prediction_complete(symbol, result):
            if result['status'] == 'success':
                logger.info(f"✓ {symbol}: {result['predicted_change_pct']:+.2f}%")

                # アラートチェック
                alert_manager.check_prediction(result)

        # バッチ完了時のコールバック
        def on_batch_complete(results):
            success_count = sum(1 for r in results.values() if r['status'] == 'success')
            logger.info(f"\n✓ Batch prediction completed: {success_count}/{len(results)} succeeded")

            # アラートサマリー
            summary = alert_manager.get_summary()
            if summary['recent_count'] > 0:
                logger.warning(f"⚠️ {summary['recent_count']} new alert(s)")

        scheduler.set_callbacks(
            on_prediction_complete=on_prediction_complete,
            on_batch_complete=on_batch_complete
        )

        # スケジューラー開始
        market_hours_only = get_config('realtime_prediction.market_hours_only', True)
        scheduler.start(symbols=symbol_list, market_hours_only=market_hours_only)

        logger.info("Smart scheduler started")
        logger.info("Press Ctrl+C to stop...\n")

        try:
            # メインスレッドを維持
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\nStopping scheduler...")
            scheduler.stop()
            logger.info("Scheduler stopped")

    logger.info("\n" + "=" * 60)
    logger.info("Real-Time Prediction workflow completed!")
    logger.info("=" * 60)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='JTrading System')

    parser.add_argument(
        '--mode',
        choices=['gui', 'gui2', 'cli', 'collect', 'train', 'backtest', 'full', 'news', 'automl', 'realtime'],
        default='gui2',
        help='Execution mode (gui2: Enhanced GUI, news: News collection, automl: AutoML, realtime: Real-time prediction)'
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default='7203.T',
        help='Stock symbol (default: 7203.T - Toyota)'
    )

    parser.add_argument(
        '--realtime-mode',
        choices=['once', 'schedule', 'smart'],
        default='once',
        help='Real-time prediction mode (once: One-time, schedule: Scheduled, smart: Smart schedule)'
    )

    args = parser.parse_args()

    # システムセットアップ
    logger = setup_system()

    try:
        if args.mode == 'gui':
            # GUI モード（シンプル版）
            logger.info("Launching GUI mode...")
            launch_gui()

        elif args.mode == 'gui2':
            # GUI モード（拡張版 - チャート付き）
            logger.info("Launching Enhanced GUI mode with charts...")
            launch_enhanced_gui()

        elif args.mode == 'collect':
            # データ収集のみ
            symbols = get_symbols()
            collect_data_workflow(symbols, logger)

        elif args.mode == 'train':
            # モデル訓練のみ
            train_model_workflow(args.symbol, logger)

        elif args.mode == 'backtest':
            # バックテストのみ
            backtest_workflow(args.symbol, logger)

        elif args.mode == 'full':
            # フルワークフロー
            full_workflow(args.symbol, logger)

        elif args.mode == 'news':
            # ニュース収集・感情分析
            symbols = get_symbols()
            news_workflow(symbols, logger)

        elif args.mode == 'automl':
            # AutoML（自動機械学習）
            automl_workflow(args.symbol, logger)

        elif args.mode == 'realtime':
            # リアルタイム予測
            symbols = get_symbols('priority') or [args.symbol]
            realtime_prediction_workflow(symbols, logger, mode=args.realtime_mode)

        elif args.mode == 'cli':
            # CLI対話モード（将来実装）
            logger.info("CLI mode is not yet implemented")

    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
