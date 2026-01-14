"""
Validators Module
データバリデーション機能
"""

import pandas as pd
import numpy as np
from typing import Union, List, Optional
from datetime import datetime


class ValidationError(Exception):
    """バリデーションエラー"""
    pass


def validate_symbol(symbol: str) -> bool:
    """
    銘柄コードの妥当性を検証

    Args:
        symbol: 銘柄コード

    Returns:
        bool: 妥当性

    Raises:
        ValidationError: 不正な銘柄コード
    """
    if not isinstance(symbol, str):
        raise ValidationError(f"Symbol must be a string, got {type(symbol)}")

    if not symbol:
        raise ValidationError("Symbol cannot be empty")

    # 日本株の場合、.T サフィックスが必要
    if not symbol.endswith('.T'):
        raise ValidationError(f"Japanese stocks must end with .T: {symbol}")

    return True


def validate_date_range(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime]
) -> bool:
    """
    日付範囲の妥当性を検証

    Args:
        start_date: 開始日
        end_date: 終了日

    Returns:
        bool: 妥当性

    Raises:
        ValidationError: 不正な日付範囲
    """
    if isinstance(start_date, str):
        try:
            start_date = pd.to_datetime(start_date)
        except Exception as e:
            raise ValidationError(f"Invalid start_date format: {start_date}") from e

    if isinstance(end_date, str):
        try:
            end_date = pd.to_datetime(end_date)
        except Exception as e:
            raise ValidationError(f"Invalid end_date format: {end_date}") from e

    if start_date >= end_date:
        raise ValidationError(f"start_date must be before end_date: {start_date} >= {end_date}")

    return True


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 1
) -> bool:
    """
    データフレームの妥当性を検証

    Args:
        df: データフレーム
        required_columns: 必須カラムのリスト
        min_rows: 最小行数

    Returns:
        bool: 妥当性

    Raises:
        ValidationError: 不正なデータフレーム
    """
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(f"Expected DataFrame, got {type(df)}")

    if df.empty:
        raise ValidationError("DataFrame is empty")

    if len(df) < min_rows:
        raise ValidationError(f"DataFrame has {len(df)} rows, minimum {min_rows} required")

    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValidationError(f"Missing required columns: {missing_columns}")

    return True


def validate_price_data(df: pd.DataFrame) -> bool:
    """
    株価データの妥当性を検証

    Args:
        df: 株価データフレーム

    Returns:
        bool: 妥当性

    Raises:
        ValidationError: 不正な株価データ
    """
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    validate_dataframe(df, required_columns=required_columns)

    # 価格データの論理チェック
    if (df['High'] < df['Low']).any():
        raise ValidationError("High price cannot be lower than Low price")

    if (df['High'] < df['Close']).any():
        raise ValidationError("High price cannot be lower than Close price")

    if (df['Low'] > df['Close']).any():
        raise ValidationError("Low price cannot be higher than Close price")

    if (df['High'] < df['Open']).any():
        raise ValidationError("High price cannot be lower than Open price")

    if (df['Low'] > df['Open']).any():
        raise ValidationError("Low price cannot be higher than Open price")

    # 負の値チェック
    price_columns = ['Open', 'High', 'Low', 'Close']
    for col in price_columns:
        if (df[col] <= 0).any():
            raise ValidationError(f"{col} price must be positive")

    if (df['Volume'] < 0).any():
        raise ValidationError("Volume cannot be negative")

    return True


def validate_model_config(config: dict) -> bool:
    """
    モデル設定の妥当性を検証

    Args:
        config: モデル設定辞書

    Returns:
        bool: 妥当性

    Raises:
        ValidationError: 不正なモデル設定
    """
    if not isinstance(config, dict):
        raise ValidationError(f"Model config must be a dictionary, got {type(config)}")

    if not config:
        raise ValidationError("Model config cannot be empty")

    return True


def validate_prediction_days(days: int) -> bool:
    """
    予測日数の妥当性を検証

    Args:
        days: 予測日数

    Returns:
        bool: 妥当性

    Raises:
        ValidationError: 不正な予測日数
    """
    if not isinstance(days, int):
        raise ValidationError(f"Prediction days must be an integer, got {type(days)}")

    if days <= 0:
        raise ValidationError(f"Prediction days must be positive, got {days}")

    if days > 365:
        raise ValidationError(f"Prediction days cannot exceed 365, got {days}")

    return True


def validate_features(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray]
) -> bool:
    """
    特徴量とターゲットの妥当性を検証

    Args:
        X: 特徴量
        y: ターゲット

    Returns:
        bool: 妥当性

    Raises:
        ValidationError: 不正な特徴量またはターゲット
    """
    # サンプル数の一致確認
    n_samples_X = len(X)
    n_samples_y = len(y)

    if n_samples_X != n_samples_y:
        raise ValidationError(
            f"Number of samples mismatch: X has {n_samples_X}, y has {n_samples_y}"
        )

    # 欠損値チェック
    if isinstance(X, pd.DataFrame):
        if X.isnull().any().any():
            null_cols = X.columns[X.isnull().any()].tolist()
            raise ValidationError(f"Features contain NaN values in columns: {null_cols}")
    elif isinstance(X, np.ndarray):
        if np.isnan(X).any():
            raise ValidationError("Features contain NaN values")

    if isinstance(y, pd.Series):
        if y.isnull().any():
            raise ValidationError("Target contains NaN values")
    elif isinstance(y, np.ndarray):
        if np.isnan(y).any():
            raise ValidationError("Target contains NaN values")

    # 無限大チェック
    if isinstance(X, pd.DataFrame):
        if np.isinf(X.values).any():
            raise ValidationError("Features contain infinite values")
    elif isinstance(X, np.ndarray):
        if np.isinf(X).any():
            raise ValidationError("Features contain infinite values")

    if isinstance(y, pd.Series):
        if np.isinf(y.values).any():
            raise ValidationError("Target contains infinite values")
    elif isinstance(y, np.ndarray):
        if np.isinf(y).any():
            raise ValidationError("Target contains infinite values")

    return True
