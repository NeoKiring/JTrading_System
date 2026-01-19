"""
Sentiment Analyzer Module
感情分析モジュール（transformers使用）
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from ...utils.logger import get_logger
from ...utils.config import get_config
from ...data.storage.database import DatabaseManager, NewsArticle

logger = get_logger(__name__)

# transformersのインポート（遅延インポート）
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers library not available. Sentiment analysis will use fallback method.")


class SentimentAnalyzer:
    """感情分析クラス"""

    def __init__(self, model_name: Optional[str] = None):
        """
        初期化

        Args:
            model_name: 使用するモデル名（省略時は設定ファイルから取得）
        """
        self.model_name = model_name or get_config(
            'sentiment.model_name',
            'cardiffnlp/twitter-xlm-roberta-base-sentiment'  # 多言語対応モデル
        )
        self.device = 'cuda' if self._check_cuda() else 'cpu'
        self.model = None
        self.tokenizer = None
        self.db_manager = DatabaseManager()

        # モデルの初期化
        if TRANSFORMERS_AVAILABLE:
            self._initialize_model()
        else:
            logger.warning("Using fallback sentiment analysis (keyword-based)")

    def _check_cuda(self) -> bool:
        """CUDA利用可能性チェック"""
        if TRANSFORMERS_AVAILABLE:
            try:
                import torch
                return torch.cuda.is_available()
            except:
                return False
        return False

    def _initialize_model(self):
        """モデルとトークナイザーの初期化"""
        try:
            logger.info(f"Loading sentiment analysis model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Error loading sentiment model: {e}")
            logger.warning("Falling back to keyword-based sentiment analysis")
            self.model = None
            self.tokenizer = None

    def analyze_text(self, text: str) -> Tuple[float, str]:
        """
        テキストの感情を分析

        Args:
            text: 分析対象テキスト

        Returns:
            Tuple[float, str]: (感情スコア, 感情ラベル)
                感情スコア: -1.0（ネガティブ）～ 1.0（ポジティブ）
                感情ラベル: 'positive', 'neutral', 'negative'
        """
        if not text or not isinstance(text, str):
            return 0.0, 'neutral'

        # transformersモデルを使用
        if TRANSFORMERS_AVAILABLE and self.model is not None:
            return self._analyze_with_transformer(text)
        else:
            # フォールバック: キーワードベース
            return self._analyze_with_keywords(text)

    def _analyze_with_transformer(self, text: str) -> Tuple[float, str]:
        """
        transformersモデルで感情分析

        Args:
            text: 分析対象テキスト

        Returns:
            Tuple[float, str]: (感情スコア, 感情ラベル)
        """
        try:
            # テキストのトークン化
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 推論
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1)

            # スコアの取得
            scores = scores.cpu().numpy()[0]

            # モデルのラベルマッピング
            # cardiffnlp/twitter-xlm-roberta-base-sentiment: [negative, neutral, positive]
            if len(scores) == 3:
                negative_score = scores[0]
                neutral_score = scores[1]
                positive_score = scores[2]

                # -1.0 ～ 1.0 にスケーリング
                sentiment_score = positive_score - negative_score

                # ラベル判定
                if sentiment_score > 0.3:
                    label = 'positive'
                elif sentiment_score < -0.3:
                    label = 'negative'
                else:
                    label = 'neutral'

            else:
                # 2クラス分類の場合
                sentiment_score = scores[1] - scores[0]
                label = 'positive' if sentiment_score > 0 else 'negative'

            return float(sentiment_score), label

        except Exception as e:
            logger.error(f"Error in transformer sentiment analysis: {e}")
            return self._analyze_with_keywords(text)

    def _analyze_with_keywords(self, text: str) -> Tuple[float, str]:
        """
        キーワードベースの感情分析（フォールバック）

        Args:
            text: 分析対象テキスト

        Returns:
            Tuple[float, str]: (感情スコア, 感情ラベル)
        """
        # ポジティブキーワード
        positive_keywords = [
            '上昇', '増加', '好調', '成長', '拡大', '利益', '黒字',
            '上方修正', '過去最高', '記録的', '好決算', '強い', '高い',
            'プラス', '改善', '回復', '躍進', '好転', '伸びる'
        ]

        # ネガティブキーワード
        negative_keywords = [
            '下落', '減少', '不調', '縮小', '損失', '赤字', '下方修正',
            '不振', '低迷', '悪化', '減益', '減収', '弱い', '低い',
            'マイナス', '懸念', 'リスク', '不安', '苦戦', '停滞'
        ]

        # キーワードカウント
        positive_count = sum(1 for keyword in positive_keywords if keyword in text)
        negative_count = sum(1 for keyword in negative_keywords if keyword in text)

        # スコア計算
        total_count = positive_count + negative_count
        if total_count == 0:
            return 0.0, 'neutral'

        sentiment_score = (positive_count - negative_count) / total_count

        # ラベル判定
        if sentiment_score > 0.2:
            label = 'positive'
        elif sentiment_score < -0.2:
            label = 'negative'
        else:
            label = 'neutral'

        return float(sentiment_score), label

    def analyze_news_batch(
        self,
        news_df: pd.DataFrame,
        text_column: str = 'title'
    ) -> pd.DataFrame:
        """
        ニュースデータフレームの感情を一括分析

        Args:
            news_df: ニュースデータフレーム
            text_column: 分析対象テキストのカラム名

        Returns:
            pd.DataFrame: 感情スコア・ラベルを追加したデータフレーム
        """
        if news_df.empty:
            return news_df

        logger.info(f"Analyzing sentiment for {len(news_df)} news articles")

        sentiment_scores = []
        sentiment_labels = []

        for _, row in news_df.iterrows():
            text = row.get(text_column, '')

            # タイトルと説明を結合して分析
            if 'description' in news_df.columns and row.get('description'):
                text = f"{text} {row.get('description')}"

            score, label = self.analyze_text(text)
            sentiment_scores.append(score)
            sentiment_labels.append(label)

        news_df['sentiment_score'] = sentiment_scores
        news_df['sentiment_label'] = sentiment_labels

        logger.info(f"Sentiment analysis completed. Positive: {sentiment_labels.count('positive')}, "
                   f"Neutral: {sentiment_labels.count('neutral')}, "
                   f"Negative: {sentiment_labels.count('negative')}")

        return news_df

    def update_database_sentiment(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        データベース内のニュースの感情スコアを更新

        Args:
            symbol: 銘柄コード（省略時は全銘柄）
            start_date: 開始日（YYYY-MM-DD形式）
            end_date: 終了日（YYYY-MM-DD形式）
        """
        try:
            with self.db_manager.get_session() as session:
                # ニュース記事の取得
                query = session.query(NewsArticle)

                if symbol:
                    query = query.filter_by(symbol=symbol)
                if start_date:
                    query = query.filter(NewsArticle.published_at >= pd.to_datetime(start_date))
                if end_date:
                    query = query.filter(NewsArticle.published_at <= pd.to_datetime(end_date))

                # 感情スコアが未設定の記事のみ
                query = query.filter(NewsArticle.sentiment_score.is_(None))

                articles = query.all()

                if not articles:
                    logger.info("No articles to update")
                    return

                logger.info(f"Updating sentiment for {len(articles)} articles")

                # 感情分析と更新
                for article in articles:
                    text = f"{article.title} {article.description or ''}"
                    score, label = self.analyze_text(text)

                    article.sentiment_score = score
                    article.sentiment_label = label

                logger.info(f"Updated sentiment for {len(articles)} articles in database")

        except Exception as e:
            logger.error(f"Error updating sentiment in database: {e}")

    def get_sentiment_summary(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """
        銘柄の感情サマリーを取得

        Args:
            symbol: 銘柄コード
            start_date: 開始日（YYYY-MM-DD形式）
            end_date: 終了日（YYYY-MM-DD形式）

        Returns:
            Dict: 感情サマリー
        """
        try:
            with self.db_manager.get_session() as session:
                query = session.query(NewsArticle).filter_by(symbol=symbol)

                if start_date:
                    query = query.filter(NewsArticle.published_at >= pd.to_datetime(start_date))
                if end_date:
                    query = query.filter(NewsArticle.published_at <= pd.to_datetime(end_date))

                articles = query.all()

                if not articles:
                    return {
                        'total_articles': 0,
                        'average_sentiment': 0.0,
                        'positive_count': 0,
                        'neutral_count': 0,
                        'negative_count': 0
                    }

                scores = [a.sentiment_score for a in articles if a.sentiment_score is not None]
                labels = [a.sentiment_label for a in articles if a.sentiment_label is not None]

                return {
                    'total_articles': len(articles),
                    'average_sentiment': np.mean(scores) if scores else 0.0,
                    'positive_count': labels.count('positive'),
                    'neutral_count': labels.count('neutral'),
                    'negative_count': labels.count('negative'),
                    'positive_ratio': labels.count('positive') / len(labels) if labels else 0.0,
                    'negative_ratio': labels.count('negative') / len(labels) if labels else 0.0
                }

        except Exception as e:
            logger.error(f"Error getting sentiment summary: {e}")
            return {}
