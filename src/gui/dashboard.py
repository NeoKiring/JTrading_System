"""
Dashboard Module
ダッシュボード表示機能
"""

import tkinter as tk
from tkinter import ttk
import ttkbootstrap as ttk_boot
from ttkbootstrap.constants import *
from datetime import datetime
from typing import Dict, Optional
import pandas as pd

from ..utils.logger import get_logger
from ..utils.config import get_config, get_symbols

logger = get_logger(__name__)


class Dashboard:
    """ダッシュボードクラス"""

    def __init__(self, parent):
        """
        初期化

        Args:
            parent: 親ウィジェット
        """
        self.parent = parent
        self._create_widgets()

    def _create_widgets(self):
        """ウィジェットを作成"""
        # メインフレーム
        main_frame = ttk_boot.Frame(self.parent, padding=10)
        main_frame.pack(fill=BOTH, expand=YES)

        # タイトル
        title_label = ttk_boot.Label(
            main_frame,
            text="システムダッシュボード",
            font=("Helvetica", 16, "bold"),
            bootstyle=INFO
        )
        title_label.pack(pady=(0, 20))

        # 上部: 統計カード
        stats_frame = ttk_boot.Frame(main_frame)
        stats_frame.pack(fill=X, pady=(0, 20))

        # カード1: システム状態
        self.status_card = self._create_stat_card(
            stats_frame,
            "システム状態",
            "準備完了",
            "✓",
            SUCCESS
        )
        self.status_card.pack(side=LEFT, padx=5, fill=X, expand=YES)

        # カード2: データ状態
        self.data_card = self._create_stat_card(
            stats_frame,
            "データ状態",
            "未収集",
            "◯",
            WARNING
        )
        self.data_card.pack(side=LEFT, padx=5, fill=X, expand=YES)

        # カード3: モデル状態
        self.model_card = self._create_stat_card(
            stats_frame,
            "モデル状態",
            "未訓練",
            "◯",
            SECONDARY
        )
        self.model_card.pack(side=LEFT, padx=5, fill=X, expand=YES)

        # カード4: 最終予測
        self.prediction_card = self._create_stat_card(
            stats_frame,
            "最終予測",
            "-",
            "📊",
            INFO
        )
        self.prediction_card.pack(side=LEFT, padx=5, fill=X, expand=YES)

        # 中央: 銘柄リストと予測結果
        content_frame = ttk_boot.Frame(main_frame)
        content_frame.pack(fill=BOTH, expand=YES)

        # 左側: 銘柄リスト
        left_frame = ttk_boot.Labelframe(
            content_frame,
            text="監視銘柄",
            padding=10,
            bootstyle=PRIMARY
        )
        left_frame.pack(side=LEFT, fill=BOTH, expand=YES, padx=(0, 10))

        # 銘柄リストのツリービュー
        self.symbols_tree = self._create_symbols_tree(left_frame)
        self.symbols_tree.pack(fill=BOTH, expand=YES)

        # 右側: 予測結果
        right_frame = ttk_boot.Labelframe(
            content_frame,
            text="予測結果",
            padding=10,
            bootstyle=INFO
        )
        right_frame.pack(side=LEFT, fill=BOTH, expand=YES)

        # 予測結果のツリービュー
        self.predictions_tree = self._create_predictions_tree(right_frame)
        self.predictions_tree.pack(fill=BOTH, expand=YES)

        # 下部: アクションボタン
        action_frame = ttk_boot.Frame(main_frame)
        action_frame.pack(fill=X, pady=(20, 0))

        ttk_boot.Button(
            action_frame,
            text="データ更新",
            command=self._on_update_data,
            bootstyle=SUCCESS,
            width=15
        ).pack(side=LEFT, padx=5)

        ttk_boot.Button(
            action_frame,
            text="モデル訓練",
            command=self._on_train_model,
            bootstyle=PRIMARY,
            width=15
        ).pack(side=LEFT, padx=5)

        ttk_boot.Button(
            action_frame,
            text="予測実行",
            command=self._on_predict,
            bootstyle=INFO,
            width=15
        ).pack(side=LEFT, padx=5)

        ttk_boot.Button(
            action_frame,
            text="バックテスト",
            command=self._on_backtest,
            bootstyle=WARNING,
            width=15
        ).pack(side=LEFT, padx=5)

        # 初期データの読み込み
        self._load_initial_data()

    def _create_stat_card(
        self,
        parent,
        title: str,
        value: str,
        icon: str,
        bootstyle
    ) -> ttk_boot.Frame:
        """統計カードを作成"""
        card = ttk_boot.Labelframe(
            parent,
            text=title,
            padding=15,
            bootstyle=bootstyle
        )

        icon_label = ttk_boot.Label(
            card,
            text=icon,
            font=("Helvetica", 24)
        )
        icon_label.pack()

        value_label = ttk_boot.Label(
            card,
            text=value,
            font=("Helvetica", 14, "bold")
        )
        value_label.pack()

        # 値を更新できるようにラベルを保存
        card.value_label = value_label

        return card

    def _create_symbols_tree(self, parent) -> ttk.Treeview:
        """銘柄リストのツリービューを作成"""
        # スクロールバー付きフレーム
        tree_frame = ttk_boot.Frame(parent)
        tree_frame.pack(fill=BOTH, expand=YES)

        # ツリービュー
        columns = ("code", "name", "sector", "status")
        tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            height=10
        )

        # カラムの設定
        tree.heading("code", text="コード")
        tree.heading("name", text="銘柄名")
        tree.heading("sector", text="セクター")
        tree.heading("status", text="状態")

        tree.column("code", width=80, anchor=CENTER)
        tree.column("name", width=150)
        tree.column("sector", width=100)
        tree.column("status", width=80, anchor=CENTER)

        # スクロールバー
        scrollbar = ttk.Scrollbar(tree_frame, orient=VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side=LEFT, fill=BOTH, expand=YES)
        scrollbar.pack(side=RIGHT, fill=Y)

        return tree

    def _create_predictions_tree(self, parent) -> ttk.Treeview:
        """予測結果のツリービューを作成"""
        # スクロールバー付きフレーム
        tree_frame = ttk_boot.Frame(parent)
        tree_frame.pack(fill=BOTH, expand=YES)

        # ツリービュー
        columns = ("symbol", "current", "predicted", "change", "confidence")
        tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            height=10
        )

        # カラムの設定
        tree.heading("symbol", text="銘柄")
        tree.heading("current", text="現在価格")
        tree.heading("predicted", text="予測価格")
        tree.heading("change", text="変化率")
        tree.heading("confidence", text="信頼度")

        tree.column("symbol", width=80, anchor=CENTER)
        tree.column("current", width=100, anchor=E)
        tree.column("predicted", width=100, anchor=E)
        tree.column("change", width=80, anchor=E)
        tree.column("confidence", width=80, anchor=CENTER)

        # スクロールバー
        scrollbar = ttk.Scrollbar(tree_frame, orient=VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side=LEFT, fill=BOTH, expand=YES)
        scrollbar.pack(side=RIGHT, fill=Y)

        return tree

    def _load_initial_data(self):
        """初期データを読み込み"""
        # 銘柄リストの読み込み
        symbols = get_symbols()

        for symbol_data in symbols:
            symbol = symbol_data.get('symbol', '')
            name = symbol_data.get('name', '')
            sector = symbol_data.get('sector', '')

            self.symbols_tree.insert(
                "",
                END,
                values=(symbol, name, sector, "未取得")
            )

        logger.info(f"Loaded {len(symbols)} symbols")

    def update_status(self, status: str, data_status: str = None, model_status: str = None):
        """ステータスを更新"""
        if status:
            self.status_card.value_label.config(text=status)

        if data_status:
            self.data_card.value_label.config(text=data_status)

        if model_status:
            self.model_card.value_label.config(text=model_status)

    def update_predictions(self, predictions: Dict):
        """予測結果を更新"""
        # 既存のデータをクリア
        for item in self.predictions_tree.get_children():
            self.predictions_tree.delete(item)

        # 新しいデータを追加
        for symbol, pred_data in predictions.items():
            current = pred_data.get('current_price', 0)
            predicted = pred_data.get('predicted_price', 0)
            change = ((predicted - current) / current * 100) if current > 0 else 0
            confidence = pred_data.get('confidence', 0)

            # 変化率に応じてタグを設定
            tag = 'positive' if change > 0 else 'negative'

            self.predictions_tree.insert(
                "",
                END,
                values=(
                    symbol,
                    f"¥{current:,.0f}",
                    f"¥{predicted:,.0f}",
                    f"{change:+.2f}%",
                    f"{confidence:.1%}"
                ),
                tags=(tag,)
            )

        # タグの色設定
        self.predictions_tree.tag_configure('positive', foreground='#26a69a')
        self.predictions_tree.tag_configure('negative', foreground='#ef5350')

        logger.info(f"Updated {len(predictions)} predictions")

    # イベントハンドラ
    def _on_update_data(self):
        """データ更新"""
        logger.info("Data update requested from dashboard")
        self.update_status(status="データ更新中...")
        # 実際の処理はメインウィンドウから呼び出す

    def _on_train_model(self):
        """モデル訓練"""
        logger.info("Model training requested from dashboard")
        self.update_status(status="モデル訓練中...")

    def _on_predict(self):
        """予測実行"""
        logger.info("Prediction requested from dashboard")
        self.update_status(status="予測実行中...")

    def _on_backtest(self):
        """バックテスト"""
        logger.info("Backtest requested from dashboard")
        self.update_status(status="バックテスト実行中...")
