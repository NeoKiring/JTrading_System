"""
Database Module
SQLAlchemy ORMを使用したデータベース管理
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from datetime import datetime
from typing import Optional, List
from pathlib import Path
from contextlib import contextmanager

from ...utils.logger import get_logger
from ...utils.config import get_config

logger = get_logger(__name__)

Base = declarative_base()


class StockPrice(Base):
    """株価データモデル"""
    __tablename__ = 'stock_prices'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    created_at = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"<StockPrice(symbol='{self.symbol}', date='{self.date}', close={self.close})>"


class NewsArticle(Base):
    """ニュース記事モデル"""
    __tablename__ = 'news_articles'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), index=True)
    title = Column(Text, nullable=False)
    content = Column(Text)
    url = Column(Text)
    published_at = Column(DateTime)
    sentiment_score = Column(Float)
    created_at = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"<NewsArticle(symbol='{self.symbol}', title='{self.title[:30]}...')>"


class Prediction(Base):
    """予測結果モデル"""
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    prediction_date = Column(Date, nullable=False, index=True)
    target_date = Column(Date, nullable=False)
    predicted_price = Column(Float)
    confidence_lower = Column(Float)
    confidence_upper = Column(Float)
    model_name = Column(String(100))
    created_at = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"<Prediction(symbol='{self.symbol}', target_date='{self.target_date}', " \
               f"predicted_price={self.predicted_price})>"


class Model(Base):
    """モデル情報モデル"""
    __tablename__ = 'models'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)
    model_type = Column(String(50))
    parameters = Column(Text)  # JSON文字列として保存
    performance_metrics = Column(Text)  # JSON文字列として保存
    file_path = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    def __repr__(self):
        return f"<Model(name='{self.name}', model_type='{self.model_type}')>"


class DatabaseManager:
    """データベース管理クラス（Singletonパターン）"""

    _instance: Optional['DatabaseManager'] = None
    _engine = None
    _session_factory = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初期化"""
        if self._engine is None:
            self._initialize_database()

    def _initialize_database(self):
        """データベースの初期化"""
        # 設定から データベースパスを取得
        db_path = get_config('database.path', 'data/jtrading.db')
        db_echo = get_config('database.echo', False)

        # ディレクトリが存在することを確認
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # エンジンの作成
        connection_string = f'sqlite:///{db_path}'
        self._engine = create_engine(
            connection_string,
            echo=db_echo,
            connect_args={'check_same_thread': False},
            poolclass=StaticPool
        )

        # テーブルの作成
        Base.metadata.create_all(self._engine)

        # セッションファクトリーの作成
        self._session_factory = sessionmaker(bind=self._engine)

        logger.info(f"Database initialized at {db_path}")

    @contextmanager
    def get_session(self) -> Session:
        """
        セッションのコンテキストマネージャー

        Yields:
            Session: SQLAlchemyセッション
        """
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def create_all_tables(self):
        """全テーブルを作成"""
        Base.metadata.create_all(self._engine)
        logger.info("All tables created successfully")

    def drop_all_tables(self):
        """全テーブルを削除（注意：データが失われます）"""
        Base.metadata.drop_all(self._engine)
        logger.warning("All tables dropped")

    def reset_database(self):
        """データベースをリセット"""
        logger.warning("Resetting database...")
        self.drop_all_tables()
        self.create_all_tables()
        logger.info("Database reset complete")

    # Stock Price Operations
    def save_stock_prices(self, prices: List[dict]):
        """
        株価データを保存

        Args:
            prices: 株価データのリスト
        """
        with self.get_session() as session:
            for price_data in prices:
                # 既存のレコードをチェック
                existing = session.query(StockPrice).filter_by(
                    symbol=price_data['symbol'],
                    date=price_data['date']
                ).first()

                if existing:
                    # 更新
                    for key, value in price_data.items():
                        setattr(existing, key, value)
                else:
                    # 新規作成
                    stock_price = StockPrice(**price_data)
                    session.add(stock_price)

            logger.info(f"Saved {len(prices)} stock price records")

    def get_stock_prices(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[StockPrice]:
        """
        株価データを取得

        Args:
            symbol: 銘柄コード
            start_date: 開始日
            end_date: 終了日

        Returns:
            List[StockPrice]: 株価データのリスト
        """
        with self.get_session() as session:
            query = session.query(StockPrice).filter(StockPrice.symbol == symbol)

            if start_date:
                query = query.filter(StockPrice.date >= start_date)
            if end_date:
                query = query.filter(StockPrice.date <= end_date)

            return query.order_by(StockPrice.date).all()

    # News Operations
    def save_news_articles(self, articles: List[dict]):
        """
        ニュース記事を保存

        Args:
            articles: ニュース記事のリスト
        """
        with self.get_session() as session:
            for article_data in articles:
                article = NewsArticle(**article_data)
                session.add(article)

            logger.info(f"Saved {len(articles)} news articles")

    def get_news_articles(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[NewsArticle]:
        """
        ニュース記事を取得

        Args:
            symbol: 銘柄コード（オプション）
            start_date: 開始日
            end_date: 終了日

        Returns:
            List[NewsArticle]: ニュース記事のリスト
        """
        with self.get_session() as session:
            query = session.query(NewsArticle)

            if symbol:
                query = query.filter(NewsArticle.symbol == symbol)
            if start_date:
                query = query.filter(NewsArticle.published_at >= start_date)
            if end_date:
                query = query.filter(NewsArticle.published_at <= end_date)

            return query.order_by(NewsArticle.published_at.desc()).all()

    # Prediction Operations
    def save_predictions(self, predictions: List[dict]):
        """
        予測結果を保存

        Args:
            predictions: 予測結果のリスト
        """
        with self.get_session() as session:
            for pred_data in predictions:
                prediction = Prediction(**pred_data)
                session.add(prediction)

            logger.info(f"Saved {len(predictions)} predictions")

    def get_predictions(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Prediction]:
        """
        予測結果を取得

        Args:
            symbol: 銘柄コード
            start_date: 開始日
            end_date: 終了日

        Returns:
            List[Prediction]: 予測結果のリスト
        """
        with self.get_session() as session:
            query = session.query(Prediction).filter(Prediction.symbol == symbol)

            if start_date:
                query = query.filter(Prediction.prediction_date >= start_date)
            if end_date:
                query = query.filter(Prediction.prediction_date <= end_date)

            return query.order_by(Prediction.prediction_date.desc()).all()

    # Model Operations
    def save_model_info(self, model_info: dict):
        """
        モデル情報を保存

        Args:
            model_info: モデル情報
        """
        with self.get_session() as session:
            # 既存のモデルをチェック
            existing = session.query(Model).filter_by(name=model_info['name']).first()

            if existing:
                # 更新
                for key, value in model_info.items():
                    setattr(existing, key, value)
                existing.updated_at = datetime.now()
            else:
                # 新規作成
                model = Model(**model_info)
                session.add(model)

            logger.info(f"Saved model info: {model_info['name']}")

    def get_model_info(self, name: str) -> Optional[Model]:
        """
        モデル情報を取得

        Args:
            name: モデル名

        Returns:
            Model: モデル情報
        """
        with self.get_session() as session:
            return session.query(Model).filter_by(name=name).first()

    def get_all_models(self) -> List[Model]:
        """
        全モデル情報を取得

        Returns:
            List[Model]: モデル情報のリスト
        """
        with self.get_session() as session:
            return session.query(Model).order_by(Model.updated_at.desc()).all()


# シングルトンインスタンス
_db_manager = DatabaseManager()


def get_db_manager() -> DatabaseManager:
    """
    データベースマネージャーのインスタンスを取得

    Returns:
        DatabaseManager: データベースマネージャー
    """
    return _db_manager
