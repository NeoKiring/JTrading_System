"""
Chart Viewer Module
チャート表示機能（mplfinance使用）
"""

import tkinter as tk
from tkinter import ttk
import ttkbootstrap as ttk_boot
from ttkbootstrap.constants import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import mplfinance as mpf

from ..utils.logger import get_logger
from ..utils.config import get_config
from ..data.collectors.stock_collector import StockCollector

logger = get_logger(__name__)


class ChartViewer:
    """チャート表示クラス"""

    def __init__(self, parent):
        """
        初期化

        Args:
            parent: 親ウィジェット
        """
        self.parent = parent
        self.collector = StockCollector()
        self.current_symbol = None
        self.current_data = None

        self._create_widgets()

    def _create_widgets(self):
        """ウィジェットを作成"""
        # メインフレーム
        main_frame = ttk_boot.Frame(self.parent, padding=10)
        main_frame.pack(fill=BOTH, expand=YES)

        # コントロールパネル
        control_frame = ttk_boot.Labelframe(
            main_frame,
            text="チャート設定",
            padding=10,
            bootstyle=PRIMARY
        )
        control_frame.pack(fill=X, pady=(0, 10))

        # 銘柄選択
        ttk_boot.Label(control_frame, text="銘柄:").grid(row=0, column=0, padx=5, pady=5, sticky=W)

        self.symbol_var = tk.StringVar(value="7203.T")
        symbol_entry = ttk_boot.Entry(
            control_frame,
            textvariable=self.symbol_var,
            width=15,
            bootstyle=INFO
        )
        symbol_entry.grid(row=0, column=1, padx=5, pady=5)

        # 期間選択
        ttk_boot.Label(control_frame, text="期間:").grid(row=0, column=2, padx=5, pady=5, sticky=W)

        self.period_var = tk.StringVar(value="3ヶ月")
        period_combo = ttk_boot.Combobox(
            control_frame,
            textvariable=self.period_var,
            values=["1ヶ月", "3ヶ月", "6ヶ月", "1年", "3年", "5年", "全期間"],
            state="readonly",
            width=10,
            bootstyle=INFO
        )
        period_combo.grid(row=0, column=3, padx=5, pady=5)

        # チャートタイプ選択
        ttk_boot.Label(control_frame, text="表示:").grid(row=0, column=4, padx=5, pady=5, sticky=W)

        self.chart_type_var = tk.StringVar(value="ローソク足")
        chart_type_combo = ttk_boot.Combobox(
            control_frame,
            textvariable=self.chart_type_var,
            values=["ローソク足", "ライン", "エリア"],
            state="readonly",
            width=12,
            bootstyle=INFO
        )
        chart_type_combo.grid(row=0, column=5, padx=5, pady=5)

        # 更新ボタン
        update_btn = ttk_boot.Button(
            control_frame,
            text="チャート更新",
            command=self.update_chart,
            bootstyle=SUCCESS,
            width=15
        )
        update_btn.grid(row=0, column=6, padx=10, pady=5)

        # インジケーター選択
        ttk_boot.Label(control_frame, text="インジケーター:").grid(row=1, column=0, padx=5, pady=5, sticky=W)

        indicator_frame = ttk_boot.Frame(control_frame)
        indicator_frame.grid(row=1, column=1, columnspan=6, sticky=W, padx=5, pady=5)

        self.show_sma = tk.BooleanVar(value=True)
        self.show_volume = tk.BooleanVar(value=True)
        self.show_bb = tk.BooleanVar(value=False)
        self.show_macd = tk.BooleanVar(value=False)

        ttk_boot.Checkbutton(
            indicator_frame,
            text="移動平均",
            variable=self.show_sma,
            bootstyle="success-round-toggle"
        ).pack(side=LEFT, padx=5)

        ttk_boot.Checkbutton(
            indicator_frame,
            text="出来高",
            variable=self.show_volume,
            bootstyle="success-round-toggle"
        ).pack(side=LEFT, padx=5)

        ttk_boot.Checkbutton(
            indicator_frame,
            text="ボリンジャーバンド",
            variable=self.show_bb,
            bootstyle="success-round-toggle"
        ).pack(side=LEFT, padx=5)

        ttk_boot.Checkbutton(
            indicator_frame,
            text="MACD",
            variable=self.show_macd,
            bootstyle="success-round-toggle"
        ).pack(side=LEFT, padx=5)

        # チャート表示エリア
        chart_frame = ttk_boot.Frame(main_frame)
        chart_frame.pack(fill=BOTH, expand=YES)

        # 初期メッセージ
        self.chart_canvas = None
        self.toolbar = None

        initial_label = ttk_boot.Label(
            chart_frame,
            text="「チャート更新」ボタンをクリックしてチャートを表示",
            font=("Helvetica", 14),
            bootstyle=SECONDARY
        )
        initial_label.pack(pady=50)

    def update_chart(self):
        """チャートを更新"""
        symbol = self.symbol_var.get()
        period = self.period_var.get()

        if not symbol:
            logger.warning("Symbol is empty")
            return

        logger.info(f"Updating chart for {symbol} ({period})")

        # データ取得
        end_date = datetime.now()
        start_date = self._get_start_date(period, end_date)

        try:
            df = self.collector.collect(
                symbol=symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                save_to_db=False
            )

            if df is None or df.empty:
                logger.error(f"No data available for {symbol}")
                return

            self.current_symbol = symbol
            self.current_data = df

            # チャート描画
            self._draw_chart(df, symbol)

        except Exception as e:
            logger.error(f"Error updating chart: {e}")

    def _get_start_date(self, period: str, end_date: datetime) -> datetime:
        """期間から開始日を計算"""
        period_map = {
            "1ヶ月": timedelta(days=30),
            "3ヶ月": timedelta(days=90),
            "6ヶ月": timedelta(days=180),
            "1年": timedelta(days=365),
            "3年": timedelta(days=365 * 3),
            "5年": timedelta(days=365 * 5),
            "全期間": timedelta(days=365 * 10)  # 10年
        }

        delta = period_map.get(period, timedelta(days=90))
        return end_date - delta

    def _draw_chart(self, df: pd.DataFrame, symbol: str):
        """チャートを描画"""
        # 既存のチャートをクリア
        for widget in self.parent.winfo_children():
            if isinstance(widget, tk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, tk.Frame):
                        for subchild in child.winfo_children():
                            if isinstance(subchild, (tk.Canvas, tk.Frame)):
                                subchild.destroy()

        # データの準備
        df = df.copy()
        df = df.set_index('date')
        df.index = pd.to_datetime(df.index)

        # OHLCVデータの準備
        ohlc_data = df[['open', 'high', 'low', 'close', 'volume']].copy()
        ohlc_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        # 追加プロット用の設定
        addplot_list = []

        # 移動平均
        if self.show_sma.get():
            sma5 = ohlc_data['Close'].rolling(window=5).mean()
            sma20 = ohlc_data['Close'].rolling(window=20).mean()
            sma50 = ohlc_data['Close'].rolling(window=50).mean()

            addplot_list.append(mpf.make_addplot(sma5, color='cyan', width=1))
            addplot_list.append(mpf.make_addplot(sma20, color='orange', width=1))
            addplot_list.append(mpf.make_addplot(sma50, color='red', width=1.5))

        # ボリンジャーバンド
        if self.show_bb.get():
            sma20 = ohlc_data['Close'].rolling(window=20).mean()
            std20 = ohlc_data['Close'].rolling(window=20).std()
            upper_band = sma20 + (std20 * 2)
            lower_band = sma20 - (std20 * 2)

            addplot_list.append(mpf.make_addplot(upper_band, color='gray', linestyle='--', width=1))
            addplot_list.append(mpf.make_addplot(lower_band, color='gray', linestyle='--', width=1))

        # チャートタイプの決定
        chart_type_map = {
            "ローソク足": "candle",
            "ライン": "line",
            "エリア": "line"
        }
        plot_type = chart_type_map.get(self.chart_type_var.get(), "candle")

        # スタイル設定
        mc = mpf.make_marketcolors(
            up='#26a69a',
            down='#ef5350',
            edge='inherit',
            wick='inherit',
            volume='in',
            alpha=0.9
        )

        style = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle='--',
            y_on_right=False,
            facecolor='#1e1e1e',
            figcolor='#1e1e1e',
            edgecolor='#464646',
            gridcolor='#464646'
        )

        # フィギュアの作成
        fig, axes = mpf.plot(
            ohlc_data,
            type=plot_type,
            style=style,
            title=f'\n{symbol} - 株価チャート',
            ylabel='価格 (円)',
            volume=self.show_volume.get(),
            addplot=addplot_list if addplot_list else None,
            figsize=(12, 8),
            returnfig=True,
            datetime_format='%Y-%m-%d',
            xrotation=15
        )

        # Tkinterに埋め込み
        chart_frame = ttk_boot.Frame(self.parent)
        chart_frame.pack(fill=BOTH, expand=YES, pady=(0, 10))

        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=YES)

        # ツールバーの追加
        toolbar_frame = ttk_boot.Frame(self.parent)
        toolbar_frame.pack(fill=X)

        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        self.chart_canvas = canvas
        self.toolbar = toolbar

        logger.info(f"Chart updated for {symbol}")
