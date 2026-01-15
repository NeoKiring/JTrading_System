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
from src.data.processors.data_preprocessor import DataPreprocessor
from src.models.ml_models import XGBoostModel, evaluate_model
from src.backtesting.backtest_engine import BacktestEngine
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
        scale_features=True
    )

    # 学習/テストデータ分割
    test_size = 1 - get_config('model.train_test_split', 0.8)
    X_train, X_test, y_train, y_test = preprocessor.train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    # モデル訓練
    model = XGBoostModel()
    model.train(X_train, y_train, X_test, y_test)

    # モデル評価
    metrics = evaluate_model(model, X_test, y_test)

    logger.info("Model Training Results:")
    logger.info(f"  RMSE: {metrics['rmse']:.2f}")
    logger.info(f"  MAE: {metrics['mae']:.2f}")
    logger.info(f"  R²: {metrics['r2']:.4f}")
    logger.info(f"  MAPE: {metrics['mape']:.2f}%")

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
        prediction_days=prediction_days
    )

    # モデル訓練（簡易版）
    test_size = 1 - get_config('model.train_test_split', 0.8)
    X_train, X_test, y_train, y_test = preprocessor.train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

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

    return results


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


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='JTrading System')

    parser.add_argument(
        '--mode',
        choices=['gui', 'gui2', 'cli', 'collect', 'train', 'backtest', 'full'],
        default='gui2',
        help='Execution mode (gui2: Enhanced GUI with charts)'
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default='7203.T',
        help='Stock symbol (default: 7203.T - Toyota)'
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
