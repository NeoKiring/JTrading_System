"""
News Viewer Module
ニュース表示・感情分析タブ
"""

import tkinter as tk
from tkinter import ttk, messagebox
import ttkbootstrap as ttkb
from typing import Optional
import pandas as pd
from datetime import datetime, timedelta
import webbrowser
from ..utils.logger import get_logger
from ..utils.config import get_config
from ..data.collectors.news_collector import NewsCollector
from ..data.processors.sentiment_analyzer import SentimentAnalyzer

logger = get_logger(__name__)


class NewsViewer(ttkb.Frame):
    """ニュース表示・感情分析タブ"""

    def __init__(self, parent):
        """
        初期化

        Args:
            parent: 親ウィジェット
        """
        super().__init__(parent)

        self.news_collector = NewsCollector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.current_news_df = pd.DataFrame()

        self._create_widgets()

    def _create_widgets(self):
        """ウィジェットの作成"""
        # メインフレーム
        main_frame = ttkb.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # コントロールパネル
        control_frame = ttkb.LabelFrame(main_frame, text="ニュース取得設定", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # 銘柄選択
        symbol_frame = ttkb.Frame(control_frame)
        symbol_frame.pack(fill=tk.X, pady=2)

        ttkb.Label(symbol_frame, text="銘柄:").pack(side=tk.LEFT, padx=5)

        self.symbol_var = tk.StringVar()
        self.symbol_combo = ttkb.Combobox(
            symbol_frame,
            textvariable=self.symbol_var,
            width=15,
            state='readonly'
        )
        self.symbol_combo.pack(side=tk.LEFT, padx=5)

        # 銘柄リストの読み込み
        self._load_symbols()

        # 期間選択
        period_frame = ttkb.Frame(control_frame)
        period_frame.pack(fill=tk.X, pady=2)

        ttkb.Label(period_frame, text="期間:").pack(side=tk.LEFT, padx=5)

        self.period_var = tk.StringVar(value="7days")
        periods = [
            ("過去1日", "1days"),
            ("過去3日", "3days"),
            ("過去7日", "7days"),
            ("過去14日", "14days"),
            ("過去30日", "30days")
        ]

        for text, value in periods:
            rb = ttkb.Radiobutton(
                period_frame,
                text=text,
                variable=self.period_var,
                value=value
            )
            rb.pack(side=tk.LEFT, padx=5)

        # ボタンフレーム
        button_frame = ttkb.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=5)

        ttkb.Button(
            button_frame,
            text="ニュース取得",
            command=self._fetch_news,
            bootstyle="primary"
        ).pack(side=tk.LEFT, padx=5)

        ttkb.Button(
            button_frame,
            text="感情分析実行",
            command=self._analyze_sentiment,
            bootstyle="info"
        ).pack(side=tk.LEFT, padx=5)

        ttkb.Button(
            button_frame,
            text="データベースから読込",
            command=self._load_from_db,
            bootstyle="secondary"
        ).pack(side=tk.LEFT, padx=5)

        # サマリーパネル
        summary_frame = ttkb.LabelFrame(main_frame, text="感情分析サマリー", padding=10)
        summary_frame.pack(fill=tk.X, padx=5, pady=5)

        # サマリー情報表示
        self.summary_labels = {}
        summary_items = [
            ('total', '総記事数:'),
            ('avg_sentiment', '平均感情スコア:'),
            ('positive', 'ポジティブ:'),
            ('neutral', '中立:'),
            ('negative', 'ネガティブ:')
        ]

        for key, label_text in summary_items:
            frame = ttkb.Frame(summary_frame)
            frame.pack(fill=tk.X, pady=2)

            ttkb.Label(frame, text=label_text, width=15).pack(side=tk.LEFT)
            label = ttkb.Label(frame, text="-", font=('', 10, 'bold'))
            label.pack(side=tk.LEFT, padx=10)
            self.summary_labels[key] = label

        # ニュースリストフレーム
        news_frame = ttkb.LabelFrame(main_frame, text="ニュース一覧", padding=10)
        news_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # TreeView（ニュースリスト）
        tree_frame = ttkb.Frame(news_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        # スクロールバー
        scrollbar = ttkb.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # TreeView
        columns = ('date', 'source', 'title', 'sentiment', 'score')
        self.news_tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show='tree headings',
            yscrollcommand=scrollbar.set
        )
        scrollbar.config(command=self.news_tree.yview)

        # カラム設定
        self.news_tree.heading('#0', text='#')
        self.news_tree.heading('date', text='日付')
        self.news_tree.heading('source', text='ソース')
        self.news_tree.heading('title', text='タイトル')
        self.news_tree.heading('sentiment', text='感情')
        self.news_tree.heading('score', text='スコア')

        self.news_tree.column('#0', width=50)
        self.news_tree.column('date', width=100)
        self.news_tree.column('source', width=120)
        self.news_tree.column('title', width=400)
        self.news_tree.column('sentiment', width=80)
        self.news_tree.column('score', width=80)

        self.news_tree.pack(fill=tk.BOTH, expand=True)

        # ダブルクリックでURLを開く
        self.news_tree.bind('<Double-1>', self._open_article_url)

        # 詳細表示フレーム
        detail_frame = ttkb.LabelFrame(main_frame, text="記事詳細", padding=10)
        detail_frame.pack(fill=tk.BOTH, padx=5, pady=5)

        # テキストウィジェット（スクロール付き）
        detail_scroll = ttkb.Scrollbar(detail_frame)
        detail_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.detail_text = tk.Text(
            detail_frame,
            wrap=tk.WORD,
            height=6,
            yscrollcommand=detail_scroll.set
        )
        self.detail_text.pack(fill=tk.BOTH, expand=True)
        detail_scroll.config(command=self.detail_text.yview)

        # TreeViewの選択イベント
        self.news_tree.bind('<<TreeviewSelect>>', self._on_article_select)

    def _load_symbols(self):
        """銘柄リストの読み込み"""
        try:
            symbols_config = get_config('symbols', {})
            symbols_list = []

            for category, stocks in symbols_config.items():
                if isinstance(stocks, list):
                    for stock in stocks:
                        if isinstance(stock, dict) and 'symbol' in stock:
                            symbols_list.append(stock['symbol'])

            self.symbol_combo['values'] = symbols_list

            if symbols_list:
                self.symbol_combo.current(0)

        except Exception as e:
            logger.error(f"Error loading symbols: {e}")

    def _fetch_news(self):
        """ニュースを取得"""
        symbol = self.symbol_var.get()
        if not symbol:
            messagebox.showwarning("警告", "銘柄を選択してください")
            return

        # 期間の計算
        period = self.period_var.get()
        days = int(period.replace('days', ''))
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        try:
            logger.info(f"Fetching news for {symbol}")

            # ニュース取得
            self.current_news_df = self.news_collector.collect(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )

            if self.current_news_df.empty:
                messagebox.showinfo("情報", "ニュースが見つかりませんでした")
                return

            # 感情分析を実行
            self.current_news_df = self.sentiment_analyzer.analyze_news_batch(
                self.current_news_df
            )

            # 表示を更新
            self._update_news_list()
            self._update_summary()

            messagebox.showinfo("成功", f"{len(self.current_news_df)}件のニュースを取得しました")

        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            messagebox.showerror("エラー", f"ニュース取得エラー: {e}")

    def _load_from_db(self):
        """データベースからニュースを読み込み"""
        symbol = self.symbol_var.get()
        if not symbol:
            messagebox.showwarning("警告", "銘柄を選択してください")
            return

        # 期間の計算
        period = self.period_var.get()
        days = int(period.replace('days', ''))
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        try:
            logger.info(f"Loading news from database for {symbol}")

            self.current_news_df = self.news_collector.get_news_from_database(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )

            if self.current_news_df.empty:
                messagebox.showinfo("情報", "データベースにニュースが見つかりませんでした")
                return

            # 表示を更新
            self._update_news_list()
            self._update_summary()

            messagebox.showinfo("成功", f"{len(self.current_news_df)}件のニュースを読み込みました")

        except Exception as e:
            logger.error(f"Error loading news from database: {e}")
            messagebox.showerror("エラー", f"データベース読込エラー: {e}")

    def _analyze_sentiment(self):
        """現在のニュースに対して感情分析を実行"""
        if self.current_news_df.empty:
            messagebox.showwarning("警告", "ニュースデータがありません")
            return

        try:
            logger.info("Analyzing sentiment for current news")

            self.current_news_df = self.sentiment_analyzer.analyze_news_batch(
                self.current_news_df
            )

            # データベースに保存（sentiment_scoreを更新）
            symbol = self.symbol_var.get()
            if symbol:
                self.sentiment_analyzer.update_database_sentiment(symbol=symbol)

            # 表示を更新
            self._update_news_list()
            self._update_summary()

            messagebox.showinfo("成功", "感情分析が完了しました")

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            messagebox.showerror("エラー", f"感情分析エラー: {e}")

    def _update_news_list(self):
        """ニュースリストを更新"""
        # TreeViewをクリア
        for item in self.news_tree.get_children():
            self.news_tree.delete(item)

        if self.current_news_df.empty:
            return

        # データを追加
        for idx, row in self.current_news_df.iterrows():
            # 感情ラベルの色分け
            sentiment_label = row.get('sentiment_label', 'neutral')
            sentiment_score = row.get('sentiment_score', 0)

            # タグの設定
            tag = sentiment_label

            # 日付のフォーマット
            date_str = ""
            if 'published_at' in row and pd.notna(row['published_at']):
                date_str = pd.to_datetime(row['published_at']).strftime('%Y-%m-%d')

            self.news_tree.insert(
                '',
                tk.END,
                text=str(idx + 1),
                values=(
                    date_str,
                    row.get('source', ''),
                    row.get('title', ''),
                    sentiment_label,
                    f"{sentiment_score:.2f}" if pd.notna(sentiment_score) else "-"
                ),
                tags=(tag,)
            )

        # タグの色設定
        self.news_tree.tag_configure('positive', foreground='green')
        self.news_tree.tag_configure('negative', foreground='red')
        self.news_tree.tag_configure('neutral', foreground='gray')

    def _update_summary(self):
        """サマリー情報を更新"""
        if self.current_news_df.empty:
            for label in self.summary_labels.values():
                label.config(text="-")
            return

        total = len(self.current_news_df)
        avg_sentiment = self.current_news_df['sentiment_score'].mean() if 'sentiment_score' in self.current_news_df.columns else 0
        positive = len(self.current_news_df[self.current_news_df['sentiment_label'] == 'positive']) if 'sentiment_label' in self.current_news_df.columns else 0
        neutral = len(self.current_news_df[self.current_news_df['sentiment_label'] == 'neutral']) if 'sentiment_label' in self.current_news_df.columns else 0
        negative = len(self.current_news_df[self.current_news_df['sentiment_label'] == 'negative']) if 'sentiment_label' in self.current_news_df.columns else 0

        self.summary_labels['total'].config(text=f"{total}件")
        self.summary_labels['avg_sentiment'].config(text=f"{avg_sentiment:.3f}")
        self.summary_labels['positive'].config(text=f"{positive}件 ({positive/total*100:.1f}%)")
        self.summary_labels['neutral'].config(text=f"{neutral}件 ({neutral/total*100:.1f}%)")
        self.summary_labels['negative'].config(text=f"{negative}件 ({negative/total*100:.1f}%)")

    def _on_article_select(self, event):
        """記事選択時の処理"""
        selection = self.news_tree.selection()
        if not selection:
            return

        item = selection[0]
        index = int(self.news_tree.item(item, 'text')) - 1

        if index < 0 or index >= len(self.current_news_df):
            return

        article = self.current_news_df.iloc[index]

        # 詳細テキストを更新
        self.detail_text.delete('1.0', tk.END)

        detail_text = f"【タイトル】\n{article.get('title', '')}\n\n"
        detail_text += f"【説明】\n{article.get('description', '')}\n\n"
        detail_text += f"【URL】\n{article.get('url', '')}\n\n"

        if 'content' in article and article.get('content'):
            detail_text += f"【本文】\n{article.get('content', '')}\n"

        self.detail_text.insert('1.0', detail_text)

    def _open_article_url(self, event):
        """記事URLをブラウザで開く"""
        selection = self.news_tree.selection()
        if not selection:
            return

        item = selection[0]
        index = int(self.news_tree.item(item, 'text')) - 1

        if index < 0 or index >= len(self.current_news_df):
            return

        article = self.current_news_df.iloc[index]
        url = article.get('url', '')

        if url:
            webbrowser.open(url)
