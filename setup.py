"""
Setup Script
JTrading System セットアップスクリプト
"""

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.storage.database import get_db_manager
from src.utils.logger import setup_logger, get_logger


def init_database():
    """データベースを初期化"""
    print("Initializing database...")

    setup_logger()
    logger = get_logger(__name__)

    db_manager = get_db_manager()
    db_manager.create_all_tables()

    logger.info("Database initialized successfully")
    print("✓ Database initialized successfully")


def reset_database():
    """データベースをリセット"""
    response = input("WARNING: This will delete all data. Continue? (yes/no): ")

    if response.lower() != 'yes':
        print("Operation cancelled")
        return

    print("Resetting database...")

    setup_logger()
    logger = get_logger(__name__)

    db_manager = get_db_manager()
    db_manager.reset_database()

    logger.info("Database reset successfully")
    print("✓ Database reset successfully")


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description='JTrading System Setup')

    parser.add_argument(
        'command',
        choices=['init_db', 'reset_db'],
        help='Setup command'
    )

    args = parser.parse_args()

    if args.command == 'init_db':
        init_database()
    elif args.command == 'reset_db':
        reset_database()


if __name__ == "__main__":
    main()
