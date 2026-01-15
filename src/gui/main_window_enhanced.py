"""
Enhanced Main Window Module
拡張メインウィンドウ（タブ形式）
"""

import tkinter as tk
from tkinter import ttk, messagebox
import ttkbootstrap as ttk_boot
from ttkbootstrap.constants import *
from .dashboard import Dashboard
from .chart_viewer import ChartViewer
from ..utils.logger import get_logger, setup_logger
from ..utils.config import get_config

logger = get_logger(__name__)


class EnhancedMainWindow:
    """拡張メインウィンドウクラス"""

    def __init__(self):
        """初期化"""
        # テーマの取得
        theme = get_config('gui.theme', 'darkly')

        # メインウィンドウの作成
        self.root = ttk_boot.Window(themename=theme)
        self.root.title(f"{get_config('app.name', 'JTrading System')} v{get_config('app.version', '0.1.0')}")

        # ウィンドウサイズの設定
        width = get_config('gui.window_size.width', 1280)
        height = get_config('gui.window_size.height', 800)
        self.root.geometry(f"{width}x{height}")

        # 最小サイズの設定
        self.root.minsize(1024, 600)

        # UI要素の初期化
        self._create_menu()
        self._create_widgets()

        logger.info("Enhanced main window initialized")

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
        data_menu.add_command(label="データ収集", command=self._on_collect_data)
        data_menu.add_command(label="データ更新", command=self._on_update_data)
        data_menu.add_command(label="履歴表示", command=self._on_show_history)

        # モデルメニュー
        model_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="モデル", menu=model_menu)
        model_menu.add_command(label="モデル訓練", command=self._on_train_model)
        model_menu.add_command(label="モデル評価", command=self._on_evaluate_model)
        model_menu.add_command(label="予測実行", command=self._on_predict)

        # バックテストメニュー
        backtest_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="バックテスト", menu=backtest_menu)
        backtest_menu.add_command(label="バックテスト実行", command=self._on_run_backtest)
        backtest_menu.add_command(label="レポート表示", command=self._on_show_report)

        # 表示メニュー
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="表示", menu=view_menu)
        view_menu.add_command(label="ダッシュボード", command=lambda: self.notebook.select(0))
        view_menu.add_command(label="チャート", command=lambda: self.notebook.select(1))
        view_menu.add_separator()
        view_menu.add_command(label="テーマ変更", command=self._on_change_theme)

        # ヘルプメニュー
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ヘルプ", menu=help_menu)
        help_menu.add_command(label="ヘルプ", command=self._on_help)
        help_menu.add_command(label="バージョン情報", command=self._on_about)

    def _create_widgets(self):
        """ウィジェットを作成"""
        # メインコンテナ
        main_container = ttk_boot.Frame(self.root)
        main_container.pack(fill=BOTH, expand=YES)

        # ステータスバー（上部）
        self.status_bar = ttk_boot.Frame(main_container, padding=5)
        self.status_bar.pack(fill=X, side=TOP)

        self.status_label = ttk_boot.Label(
            self.status_bar,
            text="準備完了",
            font=("Helvetica", 10),
            bootstyle=INFO
        )
        self.status_label.pack(side=LEFT)

        # 時刻表示
        self.time_label = ttk_boot.Label(
            self.status_bar,
            text="",
            font=("Helvetica", 10),
            bootstyle=SECONDARY
        )
        self.time_label.pack(side=RIGHT)
        self._update_time()

        # タブノートブック
        self.notebook = ttk_boot.Notebook(main_container, bootstyle=INFO)
        self.notebook.pack(fill=BOTH, expand=YES, padx=5, pady=5)

        # タブ1: ダッシュボード
        dashboard_tab = ttk_boot.Frame(self.notebook)
        self.notebook.add(dashboard_tab, text="📊 ダッシュボード")

        self.dashboard = Dashboard(dashboard_tab)

        # タブ2: チャート
        chart_tab = ttk_boot.Frame(self.notebook)
        self.notebook.add(chart_tab, text="📈 チャート")

        self.chart_viewer = ChartViewer(chart_tab)

        # タブ3: 予測
        prediction_tab = ttk_boot.Frame(self.notebook)
        self.notebook.add(prediction_tab, text="🔮 予測")

        self._create_prediction_tab(prediction_tab)

        # タブ4: バックテスト
        backtest_tab = ttk_boot.Frame(self.notebook)
        self.notebook.add(backtest_tab, text="⚡ バックテスト")

        self._create_backtest_tab(backtest_tab)

        # タブ5: ログ
        log_tab = ttk_boot.Frame(self.notebook)
        self.notebook.add(log_tab, text="📝 ログ")

        self._create_log_tab(log_tab)

    def _create_prediction_tab(self, parent):
        """予測タブを作成"""
        frame = ttk_boot.Frame(parent, padding=20)
        frame.pack(fill=BOTH, expand=YES)

        label = ttk_boot.Label(
            frame,
            text="予測機能",
            font=("Helvetica", 18, "bold"),
            bootstyle=INFO
        )
        label.pack(pady=20)

        info_label = ttk_boot.Label(
            frame,
            text="予測機能は今後実装予定です\nメインプログラムから実行してください",
            font=("Helvetica", 12),
            bootstyle=SECONDARY
        )
        info_label.pack(pady=10)

    def _create_backtest_tab(self, parent):
        """バックテストタブを作成"""
        frame = ttk_boot.Frame(parent, padding=20)
        frame.pack(fill=BOTH, expand=YES)

        label = ttk_boot.Label(
            frame,
            text="バックテスト機能",
            font=("Helvetica", 18, "bold"),
            bootstyle=WARNING
        )
        label.pack(pady=20)

        info_label = ttk_boot.Label(
            frame,
            text="バックテスト機能は今後実装予定です\nメインプログラムから実行してください",
            font=("Helvetica", 12),
            bootstyle=SECONDARY
        )
        info_label.pack(pady=10)

    def _create_log_tab(self, parent):
        """ログタブを作成"""
        frame = ttk_boot.Frame(parent, padding=10)
        frame.pack(fill=BOTH, expand=YES)

        # ログテキストエリア
        self.log_text = tk.Text(
            frame,
            wrap=tk.WORD,
            font=("Courier", 10),
            height=25
        )
        self.log_text.pack(fill=BOTH, expand=YES)

        # スクロールバー
        scrollbar = ttk.Scrollbar(self.log_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.log_text.yview)

        self._add_log("システムを起動しました")
        self._add_log("Phase 2機能が有効化されています")

    def _add_log(self, message: str):
        """ログを追加"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    def _update_status(self, message: str):
        """ステータスを更新"""
        self.status_label.config(text=message)
        self._add_log(message)

    def _update_time(self):
        """時刻を更新"""
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=current_time)
        self.root.after(1000, self._update_time)

    # メニューコマンド
    def _on_settings(self):
        """設定"""
        messagebox.showinfo("設定", "設定画面は今後実装予定です")

    def _on_exit(self):
        """終了"""
        if messagebox.askokcancel("終了確認", "アプリケーションを終了しますか？"):
            logger.info("Application closing...")
            self.root.quit()

    def _on_collect_data(self):
        """データ収集"""
        self._update_status("データ収集を開始します...")
        messagebox.showinfo("データ収集", "データ収集機能はメインプログラムから実行してください")

    def _on_update_data(self):
        """データ更新"""
        self._update_status("データを更新中...")
        messagebox.showinfo("データ更新", "データ更新機能は今後実装予定です")

    def _on_show_history(self):
        """履歴表示"""
        messagebox.showinfo("履歴表示", "履歴表示機能は今後実装予定です")

    def _on_train_model(self):
        """モデル訓練"""
        self._update_status("モデル訓練を開始します...")
        messagebox.showinfo("モデル訓練", "モデル訓練機能はメインプログラムから実行してください")

    def _on_evaluate_model(self):
        """モデル評価"""
        messagebox.showinfo("モデル評価", "モデル評価機能は今後実装予定です")

    def _on_predict(self):
        """予測実行"""
        self._update_status("予測を実行中...")
        messagebox.showinfo("予測実行", "予測機能はメインプログラムから実行してください")

    def _on_run_backtest(self):
        """バックテスト実行"""
        self._update_status("バックテストを実行中...")
        messagebox.showinfo("バックテスト", "バックテスト機能はメインプログラムから実行してください")

    def _on_show_report(self):
        """レポート表示"""
        messagebox.showinfo("レポート表示", "レポート表示機能は今後実装予定です")

    def _on_change_theme(self):
        """テーマ変更"""
        themes = ['darkly', 'flatly', 'cosmo', 'journal', 'litera', 'lumen', 'minty', 'pulse', 'sandstone', 'united', 'yeti']

        dialog = tk.Toplevel(self.root)
        dialog.title("テーマ選択")
        dialog.geometry("300x400")

        ttk_boot.Label(dialog, text="テーマを選択してください", font=("Helvetica", 12, "bold")).pack(pady=10)

        listbox = tk.Listbox(dialog, font=("Helvetica", 10))
        for theme in themes:
            listbox.insert(tk.END, theme)
        listbox.pack(fill=BOTH, expand=YES, padx=20, pady=10)

        def apply_theme():
            selection = listbox.curselection()
            if selection:
                selected_theme = listbox.get(selection[0])
                messagebox.showinfo("テーマ変更", f"テーマ変更機能は次回起動時に反映されます\n選択: {selected_theme}")
                dialog.destroy()

        ttk_boot.Button(dialog, text="適用", command=apply_theme, bootstyle=SUCCESS).pack(pady=10)

    def _on_help(self):
        """ヘルプ"""
        help_text = """
JTrading System - 日本株式AI予測システム (Phase 2)

【主な機能】
✓ ダッシュボード: システム状態と銘柄監視
✓ チャート表示: ローソク足、移動平均、ボリンジャーバンド等
✓ データ収集: 日経225銘柄の株価データを自動収集
✓ モデル訓練: XGBoost等の機械学習モデルで予測モデルを構築
✓ 予測実行: 1週間後の株価を予測
✓ バックテスト: 過去データで戦略の有効性を検証

【Phase 2新機能】
- インタラクティブチャート表示
- ダッシュボード機能
- 複数インジケーター対応

【使用方法】
1. ダッシュボードでシステム状態を確認
2. チャートで銘柄の動きを可視化
3. メインプログラムでデータ収集・訓練・予測を実行

詳細はREADME.mdを参照してください。
        """
        messagebox.showinfo("ヘルプ", help_text)

    def _on_about(self):
        """バージョン情報"""
        version = get_config('app.version', '0.1.0')
        about_text = f"""
JTrading System
バージョン: {version} (Phase 2)

日本株式市場における個別銘柄のチャート分析と
ニュース記事分析を組み合わせた、
機械学習による先行指標モデル構築システム

【Phase 2新機能】
✓ ダッシュボード表示
✓ インタラクティブチャート
✓ 複数インジケーター対応
✓ タブ型UI

Copyright © 2026
        """
        messagebox.showinfo("バージョン情報", about_text)

    def run(self):
        """アプリケーションを実行"""
        logger.info("Starting enhanced main window...")
        self.root.mainloop()


def launch_enhanced_gui():
    """拡張GUIを起動"""
    # ロガーのセットアップ
    setup_logger()

    app = EnhancedMainWindow()
    app.run()
