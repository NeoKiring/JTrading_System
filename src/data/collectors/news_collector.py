"""
News Collector Module
ニュース収集モジュール（NewsAPI連携）
"""

import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
from .base_collector import BaseCollector
from ...utils.logger import get_logger
from ...utils.config import get_config
from ...data.storage.database import DatabaseManager, NewsArticle

logger = get_logger(__name__)


class NewsCollector(BaseCollector):
    """ニュース収集クラス（NewsAPI対応）"""

    def __init__(self, api_key: Optional[str] = None):
        """
        初期化

        Args:
            api_key: NewsAPI APIキー（省略時は設定ファイルから取得）
        """
        super().__init__()
        self.api_key = api_key or get_config('news.api_key')
        self.base_url = "https://newsapi.org/v2/everything"
        self.db_manager = DatabaseManager()

        if not self.api_key:
            logger.warning("NewsAPI key not configured. Please set 'news.api_key' in config.yaml")

    def collect(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        language: str = 'ja',
        max_articles: int = 100
    ) -> pd.DataFrame:
        """
        指定銘柄のニュースを収集

        Args:
            symbol: 銘柄コード
            start_date: 開始日（YYYY-MM-DD形式、省略時は7日前）
            end_date: 終了日（YYYY-MM-DD形式、省略時は今日）
            language: 言語（ja, en等）
            max_articles: 取得する最大記事数

        Returns:
            pd.DataFrame: ニュース記事データ
        """
        if not self.api_key:
            logger.error("NewsAPI key is not configured")
            return pd.DataFrame()

        # デフォルト日付設定
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        logger.info(f"Collecting news for {symbol} from {start_date} to {end_date}")

        # 銘柄名の取得（設定ファイルから）
        symbols_config = get_config('symbols', {})
        company_name = None
        for category in symbols_config.values():
            if isinstance(category, list):
                for stock in category:
                    if isinstance(stock, dict) and stock.get('symbol') == symbol:
                        company_name = stock.get('name')
                        break
            if company_name:
                break

        if not company_name:
            logger.warning(f"Company name not found for symbol {symbol}, using symbol as query")
            company_name = symbol

        # ニュース取得
        articles = self._fetch_news(
            query=company_name,
            from_date=start_date,
            to_date=end_date,
            language=language,
            max_articles=max_articles
        )

        if not articles:
            logger.warning(f"No news articles found for {symbol}")
            return pd.DataFrame()

        # データフレームに変換
        df = pd.DataFrame(articles)
        df['symbol'] = symbol

        # データベースに保存
        self._save_to_database(df, symbol)

        logger.info(f"Collected {len(df)} news articles for {symbol}")

        return df

    def _fetch_news(
        self,
        query: str,
        from_date: str,
        to_date: str,
        language: str = 'ja',
        max_articles: int = 100
    ) -> List[Dict]:
        """
        NewsAPIからニュースを取得

        Args:
            query: 検索クエリ
            from_date: 開始日
            to_date: 終了日
            language: 言語
            max_articles: 最大記事数

        Returns:
            List[Dict]: 記事リスト
        """
        params = {
            'q': query,
            'from': from_date,
            'to': to_date,
            'language': language,
            'sortBy': 'publishedAt',
            'pageSize': min(max_articles, 100),  # NewsAPIの最大は100
            'apiKey': self.api_key
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data['status'] != 'ok':
                logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return []

            articles = []
            for article in data.get('articles', []):
                # 必要な情報を抽出
                processed_article = {
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'author': article.get('author', 'Unknown'),
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'content': article.get('content', '')
                }
                articles.append(processed_article)

            return articles

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news from NewsAPI: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in news collection: {e}")
            return []

    def _save_to_database(self, df: pd.DataFrame, symbol: str):
        """
        ニュースデータをデータベースに保存

        Args:
            df: ニュースデータフレーム
            symbol: 銘柄コード
        """
        try:
            with self.db_manager.get_session() as session:
                for _, row in df.iterrows():
                    # 既存チェック（URLで重複判定）
                    existing = session.query(NewsArticle).filter_by(
                        url=row['url']
                    ).first()

                    if existing:
                        continue

                    # 新規記事を保存
                    article = NewsArticle(
                        symbol=symbol,
                        source=row['source'],
                        author=row['author'],
                        title=row['title'],
                        description=row['description'],
                        url=row['url'],
                        published_at=pd.to_datetime(row['published_at']),
                        content=row['content']
                    )
                    session.add(article)

                logger.info(f"Saved {len(df)} news articles to database")

        except Exception as e:
            logger.error(f"Error saving news to database: {e}")

    def get_news_from_database(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        データベースからニュースを取得

        Args:
            symbol: 銘柄コード
            start_date: 開始日（YYYY-MM-DD形式）
            end_date: 終了日（YYYY-MM-DD形式）

        Returns:
            pd.DataFrame: ニュース記事データ
        """
        try:
            with self.db_manager.get_session() as session:
                query = session.query(NewsArticle).filter_by(symbol=symbol)

                if start_date:
                    query = query.filter(NewsArticle.published_at >= pd.to_datetime(start_date))
                if end_date:
                    query = query.filter(NewsArticle.published_at <= pd.to_datetime(end_date))

                articles = query.order_by(NewsArticle.published_at.desc()).all()

                if not articles:
                    return pd.DataFrame()

                # データフレームに変換
                data = [{
                    'id': article.id,
                    'symbol': article.symbol,
                    'source': article.source,
                    'author': article.author,
                    'title': article.title,
                    'description': article.description,
                    'url': article.url,
                    'published_at': article.published_at,
                    'content': article.content,
                    'sentiment_score': article.sentiment_score,
                    'sentiment_label': article.sentiment_label
                } for article in articles]

                return pd.DataFrame(data)

        except Exception as e:
            logger.error(f"Error retrieving news from database: {e}")
            return pd.DataFrame()

    def update_news_batch(self, symbols: List[str], days: int = 7):
        """
        複数銘柄のニュースを一括更新

        Args:
            symbols: 銘柄コードリスト
            days: 過去何日分のニュースを取得するか
        """
        logger.info(f"Starting batch news collection for {len(symbols)} symbols")

        for symbol in symbols:
            try:
                self.collect(
                    symbol=symbol,
                    start_date=(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                    end_date=datetime.now().strftime('%Y-%m-%d')
                )
            except Exception as e:
                logger.error(f"Error collecting news for {symbol}: {e}")
                continue

        logger.info("Batch news collection completed")
