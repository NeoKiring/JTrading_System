"""
Cache Manager Module
データキャッシュ管理
"""

import pickle
import json
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timedelta
from ...utils.logger import get_logger

logger = get_logger(__name__)


class CacheManager:
    """キャッシュ管理クラス"""

    def __init__(self, cache_dir: str = "data/cache"):
        """
        初期化

        Args:
            cache_dir: キャッシュディレクトリ
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str, format: str = "pkl") -> Path:
        """
        キャッシュファイルパスを取得

        Args:
            key: キャッシュキー
            format: ファイルフォーマット（pkl, json）

        Returns:
            Path: キャッシュファイルパス
        """
        safe_key = key.replace('/', '_').replace('\\', '_')
        return self.cache_dir / f"{safe_key}.{format}"

    def save(
        self,
        key: str,
        data: Any,
        format: str = "pkl",
        ttl: Optional[int] = None
    ):
        """
        データをキャッシュに保存

        Args:
            key: キャッシュキー
            data: 保存するデータ
            format: フォーマット（pkl, json）
            ttl: 有効期限（秒）
        """
        cache_path = self._get_cache_path(key, format)

        try:
            if format == "pkl":
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'data': data,
                        'timestamp': datetime.now(),
                        'ttl': ttl
                    }, f)
            elif format == "json":
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'data': data,
                        'timestamp': datetime.now().isoformat(),
                        'ttl': ttl
                    }, f, ensure_ascii=False, indent=2)

            logger.debug(f"Cached data saved: {key}")

        except Exception as e:
            logger.error(f"Failed to save cache {key}: {e}")

    def load(self, key: str, format: str = "pkl") -> Optional[Any]:
        """
        キャッシュからデータを読み込む

        Args:
            key: キャッシュキー
            format: フォーマット（pkl, json）

        Returns:
            キャッシュされたデータ（存在しない場合はNone）
        """
        cache_path = self._get_cache_path(key, format)

        if not cache_path.exists():
            return None

        try:
            if format == "pkl":
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
            elif format == "json":
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    cache_data['timestamp'] = datetime.fromisoformat(cache_data['timestamp'])
            else:
                logger.error(f"Unknown format: {format}")
                return None

            # TTLチェック
            if cache_data['ttl']:
                elapsed = (datetime.now() - cache_data['timestamp']).total_seconds()
                if elapsed > cache_data['ttl']:
                    logger.debug(f"Cache expired: {key}")
                    self.delete(key, format)
                    return None

            logger.debug(f"Cache hit: {key}")
            return cache_data['data']

        except Exception as e:
            logger.error(f"Failed to load cache {key}: {e}")
            return None

    def delete(self, key: str, format: str = "pkl"):
        """
        キャッシュを削除

        Args:
            key: キャッシュキー
            format: フォーマット
        """
        cache_path = self._get_cache_path(key, format)

        if cache_path.exists():
            cache_path.unlink()
            logger.debug(f"Cache deleted: {key}")

    def clear_all(self):
        """全キャッシュを削除"""
        for cache_file in self.cache_dir.glob("*"):
            if cache_file.is_file():
                cache_file.unlink()

        logger.info("All cache cleared")

    def cleanup_expired(self):
        """期限切れキャッシュを削除"""
        cleaned = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)

                if cache_data['ttl']:
                    elapsed = (datetime.now() - cache_data['timestamp']).total_seconds()
                    if elapsed > cache_data['ttl']:
                        cache_file.unlink()
                        cleaned += 1
            except Exception as e:
                logger.warning(f"Failed to check cache {cache_file}: {e}")

        logger.info(f"Cleaned {cleaned} expired cache files")
