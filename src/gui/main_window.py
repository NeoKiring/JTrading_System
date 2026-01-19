"""
Main Window Module
メインウィンドウ（ttkbootstrap使用）
"""

import tkinter as tk
from tkinter import ttk, messagebox
import ttkbootstrap as ttk_boot
from ttkbootstrap.constants import *
from ..utils.logger import get_logger
from ..utils.config import get_config

logger = get_logger(__name__)


class MainWindow:
    """メインウィンドウクラス"""

    def __init__(self):
        """初期化"""
        # テーマの取得
        theme = get_config('gui.theme', 'darkly')

        # メインウィンドウの作成
        self.root = ttk_boot.Window(themename=theme)
        self.root.title(get_config('app.name', 'JTrading System'))

        # ウィンドウサイズの設定
        width = get_config('gui.window_size.width', 1280)
        height = get_config('gui.window_size.height', 800)
        self.root.geometry(f"{width}x{height}")

        # UI要素の初期化
        self._create_menu()
        self._create_widgets()

        logger.info("Main window initialized")

    def _create_menu(self):
        """メニューバーを作成"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # ファイルメニュー
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ファイル", menu=file_menu)
        file_menu.add_command(label="設定", command=self._on_settings)
        file_menu.add_separator()
        file_menu.add_command(label="終了", command=self._on_exit)

        # データメニュー
        data_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="データ", menu=data_menu)
        data_menu.add_command(label="データ更新", command=self._on_update_data)
        data_menu.add_command(label="履歴表示", command=self._on_show_history)

        # モデルメニュー
        model_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="モデル", menu=model_menu)
        model_menu.add_command(label="モデル訓練", command=self._on_train_model)
        model_menu.add_command(label="モデル評価", command=self._on_evaluate_model)

        # バックテストメニュー
        backtest_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="バックテスト", menu=backtest_menu)
        backtest_menu.add_command(label="バックテスト実行", command=self._on_run_backtest)
        backtest_menu.add_command(label="レポート表示", command=self._on_show_report)

        # ヘルプメニュー
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ヘルプ", menu=help_menu)
        help_menu.add_command(label="ヘルプ", command=self._on_help)
        help_menu.add_command(label="バージョン情報", command=self._on_about)

    def _create_widgets(self):
        """ウィジェットを作成"""
        # メインコンテナ
        main_frame = ttk_boot.Frame(self.root, padding=10)
        main_frame.pack(fill=BOTH, expand=YES)

        # タイトル
        title_label = ttk_boot.Label(
            main_frame,
            text="JTrading System - 日本株式AI予測システム",
            font=("Helvetica", 18, "bold"),
            bootstyle=INFO
        )
        title_label.pack(pady=20)

        # ステータスフレーム
        status_frame = ttk_boot.Labelframe(
            main_frame,
            text="システム状態",
            padding=15,
            bootstyle=PRIMARY
        )
        status_frame.pack(fill=X, pady=10)

        self.status_label = ttk_boot.Label(
            status_frame,
            text="準備完了",
            font=("Helvetica", 12)
        )
        self.status_label.pack()

        # 操作ボタンフレーム
        button_frame = ttk_boot.Frame(main_frame)
        button_frame.pack(pady=20)

        # データ収集ボタン
        collect_btn = ttk_boot.Button(
            button_frame,
            text="データ収集",
            command=self._on_collect_data,
            bootstyle=SUCCESS,
            width=20
        )
        collect_btn.grid(row=0, column=0, padx=10, pady=5)

        # モデル訓練ボタン
        train_btn = ttk_boot.Button(
            button_frame,
            text="モデル訓練",
            command=self._on_train_model,
            bootstyle=PRIMARY,
            width=20
        )
        train_btn.grid(row=0, column=1, padx=10, pady=5)

        # 予測実行ボタン
        predict_btn = ttk_boot.Button(
            button_frame,
            text="予測実行",
            command=self._on_predict,
            bootstyle=INFO,
            width=20
        )
        predict_btn.grid(row=1, column=0, padx=10, pady=5)

        # バックテストボタン
        backtest_btn = ttk_boot.Button(
            button_frame,
            text="バックテスト実行",
            command=self._on_run_backtest,
            bootstyle=WARNING,
            width=20
        )
        backtest_btn.grid(row=1, column=1, padx=10, pady=5)

        # ログフレーム
        log_frame = ttk_boot.Labelframe(
            main_frame,
            text="ログ",
            padding=10,
            bootstyle=SECONDARY
        )
        log_frame.pack(fill=BOTH, expand=YES, pady=10)

        # ログテキストエリア
        self.log_text = tk.Text(
            log_frame,
            height=15,
            wrap=tk.WORD,
            font=("Courier", 10)
        )
        self.log_text.pack(fill=BOTH, expand=YES)

        # スクロールバー
        scrollbar = ttk.Scrollbar(self.log_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.log_text.yview)

        self._add_log("システムを起動しました")

    def _add_log(self, message: str):
        """ログを追加"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)

    def _update_status(self, message: str):
        """ステータスを更新"""
        self.status_label.config(text=message)
        self._add_log(message)

    # メニューコマンド
    def _on_settings(self):
        """設定"""
        messagebox.showinfo("設定", "設定画面は今後実装予定です")

    def _on_exit(self):
        """終了"""
        if messagebox.askokcancel("終了確認", "アプリケーションを終了しますか？"):
            self.root.quit()

    def _on_update_data(self):
        """データ更新"""
        self._update_status("データを更新中...")
        messagebox.showinfo("データ更新", "データ更新機能は今後実装予定です")

    def _on_show_history(self):
        """履歴表示"""
        messagebox.showinfo("履歴表示", "履歴表示機能は今後実装予定です")

    def _on_evaluate_model(self):
        """モデル評価"""
        messagebox.showinfo("モデル評価", "モデル評価機能は今後実装予定です")

    def _on_show_report(self):
        """レポート表示"""
        messagebox.showinfo("レポート表示", "レポート表示機能は今後実装予定です")

    def _on_help(self):
        """ヘルプ"""
        help_text = """
JTrading System - 日本株式AI予測システム

【主な機能】
- データ収集: 日経225銘柄の株価データを自動収集
- モデル訓練: XGBoost等の機械学習モデルで予測モデルを構築
- 予測実行: 1週間後の株価を予測
- バックテスト: 過去データで戦略の有効性を検証

【使用方法】
1. 「データ収集」ボタンで株価データを収集
2. 「モデル訓練」ボタンでモデルを訓練
3. 「予測実行」ボタンで予測を実行
4. 「バックテスト実行」で戦略を検証

詳細はREADME.mdを参照してください。
        """
        messagebox.showinfo("ヘルプ", help_text)

    def _on_about(self):
        """バージョン情報"""
        version = get_config('app.version', '0.1.0')
        about_text = f"""
JTrading System
バージョン: {version}

日本株式市場における個別銘柄のチャート分析と
ニュース記事分析を組み合わせた、
機械学習による先行指標モデル構築システム

Copyright © 2026
        """
        messagebox.showinfo("バージョン情報", about_text)

    # ボタンコマンド
    def _on_collect_data(self):
        """データ収集"""
        self._update_status("データ収集を開始します...")
        messagebox.showinfo("データ収集", "データ収集機能はメインプログラムから実行してください")

    def _on_train_model(self):
        """モデル訓練"""
        self._update_status("モデル訓練を開始します...")
        messagebox.showinfo("モデル訓練", "モデル訓練機能はメインプログラムから実行してください")

    def _on_predict(self):
        """予測実行"""
        self._update_status("予測を実行中...")
        messagebox.showinfo("予測実行", "予測機能はメインプログラムから実行してください")

    def _on_run_backtest(self):
        """バックテスト実行"""
        self._update_status("バックテストを実行中...")
        messagebox.showinfo("バックテスト", "バックテスト機能はメインプログラムから実行してください")

    def run(self):
        """アプリケーションを実行"""
        logger.info("Starting main window...")
        self.root.mainloop()


def launch_gui():
    """GUIを起動"""
    app = MainWindow()
    app.run()
