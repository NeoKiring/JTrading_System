"""
Backtest Engine Module
シンプルなバックテストエンジン
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from ..utils.logger import get_logger
from ..utils.config import get_config

logger = get_logger(__name__)


class BacktestEngine:
    """バックテストエンジン"""

    def __init__(
        self,
        initial_cash: float = None,
        commission: float = None,
        slippage: float = None
    ):
        """
        初期化

        Args:
            initial_cash: 初期資金
            commission: 手数料率
            slippage: スリッページ率
        """
        self.initial_cash = initial_cash or get_config('backtesting.initial_cash', 10000000)
        self.commission = commission or get_config('backtesting.commission', 0.001)
        self.slippage = slippage or get_config('backtesting.slippage', 0.0005)

        self.cash = self.initial_cash
        self.positions = {}  # {symbol: shares}
        self.portfolio_value = []
        self.trades = []
        self.current_date = None

    def run(
        self,
        data: pd.DataFrame,
        predictions: pd.Series,
        threshold: float = 0.02
    ) -> Dict:
        """
        バックテストを実行

        Args:
            data: 株価データ（OHLCVを含む）
            predictions: 予測値（価格または変化率）
            threshold: 取引閾値（予測上昇率がこの値を超えたら買い）

        Returns:
            Dict: バックテスト結果

        Note:
            現在の実装では、以下の制限があります：
            1. 予測期間（prediction_days）を考慮した保有期間管理がない
            2. シグナルベースの売買のみ（予測期間後の自動決済なし）

            より正確なバックテストには、以下の改善が必要です：
            - 予測期間に基づく保有期間管理
            - ウォークフォワード分析の実装
        """
        logger.info("Running backtest...")

        # データのリセット
        self.cash = self.initial_cash
        self.positions = {}
        self.portfolio_value = []
        self.trades = []

        # データの前処理
        data = data.copy()
        data['prediction'] = predictions
        data['prediction_return'] = (data['prediction'] - data['close']) / data['close']

        # バックテストループ（シグナルは当日、約定は翌日）
        for idx in range(len(data) - 1):
            row = data.iloc[idx]
            next_row = data.iloc[idx + 1]

            current_price = row['close']
            pred_return = row['prediction_return']
            symbol = row.get('symbol', 'UNKNOWN')

            execution_price = next_row.get('open', next_row['close'])
            self.current_date = next_row.get('date', idx + 1)

            # シグナル判定（当日）
            if pred_return > threshold:
                self._execute_buy(symbol, execution_price)
            elif pred_return < -threshold:
                self._execute_sell(symbol, execution_price)

            # ポートフォリオ価値の記録（翌日終値ベース）
            portfolio_val = self._calculate_portfolio_value(next_row['close'])
            self.portfolio_value.append({
                'date': self.current_date,
                'value': portfolio_val,
                'cash': self.cash,
                'positions_value': portfolio_val - self.cash
            })

        # 結果の計算
        results = self._calculate_results()

        logger.info(f"Backtest completed. Final portfolio value: ¥{results['final_value']:,.0f}")

        return results

    def _execute_buy(self, symbol: str, price: float):
        """
        買い注文を実行

        Args:
            symbol: 銘柄コード
            price: 現在価格
        """
        # ポジションサイズの計算（利用可能資金の一定割合）
        position_size = get_config('backtesting.position_size', 0.1)
        max_investment = self.cash * position_size

        # スリッページを考慮した実際の買い価格
        buy_price = price * (1 + self.slippage)

        # 購入可能株数
        shares_to_buy = int(max_investment / buy_price)

        if shares_to_buy > 0 and self.cash >= shares_to_buy * buy_price:
            # 手数料を含めた総コスト
            total_cost = shares_to_buy * buy_price * (1 + self.commission)

            if total_cost <= self.cash:
                self.cash -= total_cost

                if symbol in self.positions:
                    self.positions[symbol] += shares_to_buy
                else:
                    self.positions[symbol] = shares_to_buy

                self.trades.append({
                    'date': self.current_date,
                    'symbol': symbol,
                    'action': 'BUY',
                    'shares': shares_to_buy,
                    'price': buy_price,
                    'cost': total_cost
                })

                logger.debug(f"BUY: {shares_to_buy} shares of {symbol} at ¥{buy_price:.2f}")

    def _execute_sell(self, symbol: str, price: float):
        """
        売り注文を実行

        Args:
            symbol: 銘柄コード
            price: 現在価格
        """
        if symbol not in self.positions or self.positions[symbol] <= 0:
            return

        shares_to_sell = self.positions[symbol]

        # スリッページを考慮した実際の売り価格
        sell_price = price * (1 - self.slippage)

        # 売却金額（手数料控除後）
        proceeds = shares_to_sell * sell_price * (1 - self.commission)

        self.cash += proceeds
        self.positions[symbol] = 0

        self.trades.append({
            'date': self.current_date,
            'symbol': symbol,
            'action': 'SELL',
            'shares': shares_to_sell,
            'price': sell_price,
            'proceeds': proceeds
        })

        logger.debug(f"SELL: {shares_to_sell} shares of {symbol} at ¥{sell_price:.2f}")

    def _calculate_portfolio_value(self, current_price: float) -> float:
        """
        ポートフォリオ価値を計算

        Args:
            current_price: 現在価格

        Returns:
            float: ポートフォリオ総価値
        """
        positions_value = sum(
            shares * current_price
            for shares in self.positions.values()
        )

        return self.cash + positions_value

    def _calculate_results(self) -> Dict:
        """
        バックテスト結果を計算

        Returns:
            Dict: 結果メトリクス
        """
        portfolio_df = pd.DataFrame(self.portfolio_value)

        if portfolio_df.empty:
            return {}

        final_value = portfolio_df['value'].iloc[-1]
        total_return = (final_value - self.initial_cash) / self.initial_cash

        # リターン系列
        portfolio_df['returns'] = portfolio_df['value'].pct_change()

        # シャープレシオ（年率換算）
        mean_return = portfolio_df['returns'].mean() * 252  # 252営業日
        std_return = portfolio_df['returns'].std() * np.sqrt(252)
        sharpe_ratio = mean_return / std_return if std_return != 0 else 0

        # 最大ドローダウン
        cumulative_max = portfolio_df['value'].cummax()
        drawdown = (portfolio_df['value'] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()

        # 勝率（利益が出た取引の割合）
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            buy_trades = trades_df[trades_df['action'] == 'BUY'].copy()
            sell_trades = trades_df[trades_df['action'] == 'SELL'].copy()

            if len(buy_trades) > 0 and len(sell_trades) > 0:
                profitable_trades = sum(
                    sell_trades.iloc[i]['proceeds'] > buy_trades.iloc[i]['cost']
                    for i in range(min(len(buy_trades), len(sell_trades)))
                )
                win_rate = profitable_trades / len(sell_trades)
            else:
                win_rate = 0
        else:
            win_rate = 0

        results = {
            'initial_value': self.initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'num_trades': len(self.trades),
            'portfolio_history': portfolio_df,
            'trades': trades_df
        }

        return results
