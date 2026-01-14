"""
Helper Functions Module
共通ヘルパー関数
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Union, List, Optional
from pathlib import Path


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    ディレクトリが存在することを確認し、なければ作成

    Args:
        directory: ディレクトリパス

    Returns:
        Path: ディレクトリパス
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_date(date_str: Union[str, datetime]) -> datetime:
    """
    日付文字列をdatetimeオブジェクトに変換

    Args:
        date_str: 日付文字列またはdatetimeオブジェクト

    Returns:
        datetime: datetimeオブジェクト
    """
    if isinstance(date_str, datetime):
        return date_str

    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y%m%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    raise ValueError(f"Unable to parse date: {date_str}")


def get_trading_days(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    country: str = "JP"
) -> List[datetime]:
    """
    取引日のリストを取得

    Args:
        start_date: 開始日
        end_date: 終了日
        country: 国コード（デフォルト: "JP" - 日本）

    Returns:
        List[datetime]: 取引日のリスト
    """
    start = parse_date(start_date)
    end = parse_date(end_date)

    # 簡易実装：土日を除外（祝日は考慮せず）
    # 本格実装ではpandas_market_calendarsなどを使用
    trading_days = []
    current = start

    while current <= end:
        # 土日以外を取引日とする
        if current.weekday() < 5:  # 0-4 = Mon-Fri
            trading_days.append(current)
        current += timedelta(days=1)

    return trading_days


def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """
    リターンを計算

    Args:
        prices: 価格系列
        periods: 期間（デフォルト: 1）

    Returns:
        pd.Series: リターン系列
    """
    return prices.pct_change(periods=periods)


def calculate_log_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """
    対数リターンを計算

    Args:
        prices: 価格系列
        periods: 期間（デフォルト: 1）

    Returns:
        pd.Series: 対数リターン系列
    """
    return np.log(prices / prices.shift(periods))


def normalize_data(
    data: Union[pd.Series, pd.DataFrame],
    method: str = "minmax"
) -> Union[pd.Series, pd.DataFrame]:
    """
    データを正規化

    Args:
        data: 正規化するデータ
        method: 正規化方法（"minmax", "zscore"）

    Returns:
        正規化されたデータ
    """
    if method == "minmax":
        return (data - data.min()) / (data.max() - data.min())
    elif method == "zscore":
        return (data - data.mean()) / data.std()
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def remove_outliers(
    data: pd.Series,
    method: str = "iqr",
    threshold: float = 1.5
) -> pd.Series:
    """
    外れ値を除去

    Args:
        data: データ系列
        method: 除去方法（"iqr", "zscore"）
        threshold: 閾値

    Returns:
        pd.Series: 外れ値が除去されたデータ
    """
    if method == "iqr":
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return data[(data >= lower_bound) & (data <= upper_bound)]

    elif method == "zscore":
        z_scores = np.abs((data - data.mean()) / data.std())
        return data[z_scores < threshold]

    else:
        raise ValueError(f"Unknown outlier removal method: {method}")


def fill_missing_values(
    data: pd.DataFrame,
    method: str = "ffill",
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    欠損値を補完

    Args:
        data: データフレーム
        method: 補完方法（"ffill", "bfill", "interpolate", "mean", "median"）
        limit: 補完する最大連続欠損数

    Returns:
        pd.DataFrame: 欠損値が補完されたデータ
    """
    if method == "ffill":
        return data.fillna(method='ffill', limit=limit)
    elif method == "bfill":
        return data.fillna(method='bfill', limit=limit)
    elif method == "interpolate":
        return data.interpolate(method='linear', limit=limit)
    elif method == "mean":
        return data.fillna(data.mean())
    elif method == "median":
        return data.fillna(data.median())
    else:
        raise ValueError(f"Unknown fill method: {method}")


def format_number(
    number: float,
    decimals: int = 2,
    thousands_sep: str = ","
) -> str:
    """
    数値をフォーマット

    Args:
        number: 数値
        decimals: 小数点以下桁数
        thousands_sep: 千の位の区切り文字

    Returns:
        str: フォーマットされた数値文字列
    """
    return f"{number:,.{decimals}f}".replace(",", thousands_sep)


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    パーセンテージをフォーマット

    Args:
        value: 値（0.01 = 1%）
        decimals: 小数点以下桁数

    Returns:
        str: フォーマットされたパーセンテージ文字列
    """
    return f"{value * 100:.{decimals}f}%"


def create_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int]
) -> pd.DataFrame:
    """
    ラグ特徴量を作成

    Args:
        df: データフレーム
        columns: ラグを作成する列名のリスト
        lags: ラグ期間のリスト

    Returns:
        pd.DataFrame: ラグ特徴量が追加されたデータフレーム
    """
    df_lagged = df.copy()

    for col in columns:
        for lag in lags:
            df_lagged[f"{col}_lag_{lag}"] = df[col].shift(lag)

    return df_lagged


def create_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int],
    functions: List[str] = ['mean', 'std']
) -> pd.DataFrame:
    """
    ローリング特徴量を作成

    Args:
        df: データフレーム
        columns: ローリング統計を計算する列名のリスト
        windows: ウィンドウサイズのリスト
        functions: 計算する統計量のリスト（'mean', 'std', 'min', 'max'）

    Returns:
        pd.DataFrame: ローリング特徴量が追加されたデータフレーム
    """
    df_rolling = df.copy()

    for col in columns:
        for window in windows:
            for func in functions:
                if func == 'mean':
                    df_rolling[f"{col}_rolling_mean_{window}"] = \
                        df[col].rolling(window=window).mean()
                elif func == 'std':
                    df_rolling[f"{col}_rolling_std_{window}"] = \
                        df[col].rolling(window=window).std()
                elif func == 'min':
                    df_rolling[f"{col}_rolling_min_{window}"] = \
                        df[col].rolling(window=window).min()
                elif func == 'max':
                    df_rolling[f"{col}_rolling_max_{window}"] = \
                        df[col].rolling(window=window).max()

    return df_rolling


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    安全な除算（ゼロ除算を回避）

    Args:
        a: 分子
        b: 分母
        default: ゼロ除算時のデフォルト値

    Returns:
        float: 除算結果
    """
    try:
        return a / b if b != 0 else default
    except (ZeroDivisionError, TypeError):
        return default
