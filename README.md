# JTrading System - 日本株式AI予測システム

## 概要

JTrading Systemは、日本株式市場における個別銘柄のチャート分析とニュース記事分析を組み合わせ、機械学習による先行指標モデルを構築するPythonベースのトレーディング支援システムです。実運用レベルの予測精度を目指し、バックテスト機能により戦略の有効性を検証できます。

## 主要機能

- **マルチソースデータ収集**: 株価データ（OHLCV）とニュース記事を自動収集
- **高度なテクニカル分析**: 50種類以上のテクニカル指標を計算
- **感情分析**: 日本語ニュースから市場センチメントを抽出
- **機械学習モデル**: 複数のアルゴリズム（XGBoost、LightGBM、LSTM等）に対応
- **バックテストエンジン**: 過去データでの戦略検証とパフォーマンス評価
- **直感的なGUI**: ダークモード対応のモダンなインターフェース
- **GPU加速**: RTX3090を活用した高速学習

## システムアーキテクチャ

### 6層アーキテクチャ設計

```
┌─────────────────────────────────────────────────────────┐
│                    GUI層 (Presentation)                  │
│  Dashboard | Chart Viewer | Settings | Report Viewer    │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                  バックテスト層 (Backtesting)            │
│    Backtest Engine | Performance Metrics | Report Gen   │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                     モデル層 (Model)                     │
│  ML Models | Deep Learning | Ensemble | Model Manager   │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│               データ処理層 (Data Processing)             │
│  Technical Indicators | Sentiment Analysis | Feature Eng│
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│               データ収集層 (Data Collection)             │
│   Stock Collector | News Collector | Scheduler          │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                インフラ層 (Infrastructure)               │
│     Database | Logger | Config Manager | Cache          │
└─────────────────────────────────────────────────────────┘
```

### 各層の役割

#### 1. GUI層（Presentation Layer）
- **目的**: ユーザーインターフェースの提供
- **技術**: Tkinter + ttkbootstrap（ダークモード対応）
- **機能**:
  - ダッシュボード: 予測結果、ポートフォリオ状況、アラート表示
  - チャートビューア: ローソク足チャート、テクニカル指標、予測シグナル
  - 設定画面: 銘柄選択、モデルパラメータ、データ収集設定
  - レポートビューア: バックテスト結果、パフォーマンスメトリクス

#### 2. バックテスト層（Backtesting Layer）
- **目的**: 戦略の有効性検証
- **技術**: backtrader + カスタムエンジン
- **機能**:
  - 過去データでの戦略シミュレーション
  - リスク・リターン分析
  - パフォーマンスメトリクス計算（シャープレシオ、最大ドローダウン等）
  - レポート生成（PDF/HTML）

#### 3. モデル層（Model Layer）
- **目的**: 予測モデルの構築と管理
- **技術**: scikit-learn, XGBoost, LightGBM, TensorFlow/Keras
- **機能**:
  - 機械学習モデル: XGBoost、LightGBM、Random Forest
  - ディープラーニングモデル: LSTM、GRU、Transformer
  - アンサンブル学習: スタッキング、ブレンディング
  - AutoML対応（将来拡張）

#### 4. データ処理層（Data Processing Layer）
- **目的**: 特徴量エンジニアリング
- **技術**: pandas, numpy, TA-Lib, transformers
- **機能**:
  - テクニカル指標計算: SMA、EMA、RSI、MACD、Bollinger Bands等
  - 感情分析: 日本語ニュース記事から市場センチメント抽出
  - 特徴量生成: ラグ特徴量、ローリング統計、差分特徴量
  - データ正規化・標準化

#### 5. データ収集層（Data Collection Layer）
- **目的**: 外部データの取得と保存
- **技術**: yfinance, NewsAPI, requests
- **機能**:
  - 株価データ収集: 日次・分足データ（2014年以降）
  - ニュースデータ収集: 企業ニュース、市場ニュース
  - スケジューラー: 1時間毎の自動データ更新
  - データキャッシング

#### 6. インフラ層（Infrastructure Layer）
- **目的**: システム基盤の提供
- **技術**: SQLite, logging, yaml
- **機能**:
  - データベース管理（SQLite）
  - ログ管理（オブザーバビリティ）
  - 設定管理（YAML形式）
  - エラーハンドリング

## ディレクトリ構成

```
JTrading_System/
├── src/                          # ソースコード
│   ├── data/                     # データ収集・処理モジュール
│   │   ├── collectors/           # データ収集
│   │   │   ├── __init__.py
│   │   │   ├── base_collector.py          # 基底クラス
│   │   │   ├── stock_collector.py         # 株価データ収集
│   │   │   ├── news_collector.py          # ニュース収集
│   │   │   └── scheduler.py               # スケジューラー
│   │   ├── processors/           # データ処理
│   │   │   ├── __init__.py
│   │   │   ├── technical_indicators.py    # テクニカル指標
│   │   │   ├── sentiment_analyzer.py      # 感情分析
│   │   │   ├── feature_engineer.py        # 特徴量エンジニアリング
│   │   │   └── data_preprocessor.py       # 前処理
│   │   └── storage/              # データ保存
│   │       ├── __init__.py
│   │       ├── database.py                # データベース管理
│   │       └── cache_manager.py           # キャッシュ管理
│   ├── models/                   # モデル定義
│   │   ├── __init__.py
│   │   ├── base_model.py                  # モデル基底クラス
│   │   ├── ml_models.py                   # 機械学習モデル
│   │   ├── deep_learning_models.py        # ディープラーニングモデル
│   │   ├── ensemble_models.py             # アンサンブルモデル
│   │   └── model_manager.py               # モデル管理
│   ├── backtesting/              # バックテストモジュール
│   │   ├── __init__.py
│   │   ├── backtest_engine.py             # バックテストエンジン
│   │   ├── strategy.py                    # 取引戦略
│   │   ├── performance_metrics.py         # パフォーマンス評価
│   │   └── report_generator.py            # レポート生成
│   ├── gui/                      # GUI
│   │   ├── __init__.py
│   │   ├── main_window.py                 # メインウィンドウ
│   │   ├── dashboard.py                   # ダッシュボード
│   │   ├── chart_viewer.py                # チャート表示
│   │   ├── settings.py                    # 設定画面
│   │   ├── report_viewer.py               # レポート表示
│   │   └── themes.py                      # テーマ管理
│   ├── utils/                    # ユーティリティ
│   │   ├── __init__.py
│   │   ├── logger.py                      # ログ管理
│   │   ├── config.py                      # 設定管理
│   │   ├── helpers.py                     # ヘルパー関数
│   │   └── validators.py                  # バリデーション
│   └── main.py                   # エントリーポイント
├── data/                         # データディレクトリ
│   ├── raw/                      # 生データ
│   │   ├── stocks/               # 株価データ
│   │   └── news/                 # ニュースデータ
│   ├── processed/                # 処理済みデータ
│   │   ├── features/             # 特徴量データ
│   │   └── targets/              # ターゲットデータ
│   └── models/                   # 訓練済みモデル
│       ├── ml/                   # 機械学習モデル
│       └── dl/                   # ディープラーニングモデル
├── logs/                         # ログファイル
│   ├── app.log                   # アプリケーションログ
│   ├── error.log                 # エラーログ
│   └── trading.log               # 取引ログ
├── config/                       # 設定ファイル
│   ├── config.yaml               # メイン設定
│   ├── symbols.yaml              # 銘柄リスト
│   ├── model_config.yaml         # モデル設定
│   └── api_keys.yaml             # APIキー（.gitignore対象）
├── tests/                        # テストコード
│   ├── __init__.py
│   ├── test_collectors.py        # データ収集テスト
│   ├── test_processors.py        # データ処理テスト
│   ├── test_models.py            # モデルテスト
│   └── test_backtesting.py       # バックテストテスト
├── docs/                         # ドキュメント
│   ├── architecture.md           # アーキテクチャ設計書
│   ├── api_documentation.md      # API仕様書
│   ├── user_guide.md             # ユーザーガイド
│   └── development_guide.md      # 開発ガイド
├── requirements.txt              # 依存パッケージ
├── requirements-dev.txt          # 開発用パッケージ
├── run.bat                       # Windows起動スクリプト
├── setup.py                      # セットアップスクリプト
├── .gitignore                    # Git除外設定
└── README.md                     # プロジェクト説明（本ファイル）
```

## 要件定義

### 1. 機能要件

#### 1.1 データ収集機能
- **FR-1.1**: 日経225銘柄の株価データを取得できること
- **FR-1.2**: 2014年以降の過去データを取得できること
- **FR-1.3**: 日次および分足データに対応すること
- **FR-1.4**: 企業に関連するニュース記事を取得できること
- **FR-1.5**: 1時間毎に自動でデータを更新できること
- **FR-1.6**: 銘柄リストを動的に変更できること

#### 1.2 データ処理機能
- **FR-2.1**: 50種類以上のテクニカル指標を計算できること
- **FR-2.2**: 日本語ニュース記事から感情スコアを算出できること
- **FR-2.3**: 欠損値の補完と外れ値の処理を行えること
- **FR-2.4**: 特徴量の正規化・標準化を行えること

#### 1.3 モデル構築機能
- **FR-3.1**: 複数の機械学習アルゴリズムをサポートすること
- **FR-3.2**: ディープラーニングモデル（LSTM等）に対応すること
- **FR-3.3**: GPU（RTX3090）を活用した高速学習が可能なこと
- **FR-3.4**: ハイパーパラメータの最適化が行えること
- **FR-3.5**: モデルの保存と読み込みができること

#### 1.4 予測機能
- **FR-4.1**: 1週間後の株価を予測できること
- **FR-4.2**: 将来的に1ヶ月後の予測に拡張できること
- **FR-4.3**: 複数銘柄を同時に予測できること
- **FR-4.4**: 予測の信頼区間を提示できること

#### 1.5 バックテスト機能
- **FR-5.1**: 過去データで戦略をシミュレーションできること
- **FR-5.2**: リターン、リスク、シャープレシオを計算できること
- **FR-5.3**: 最大ドローダウンを算出できること
- **FR-5.4**: バックテスト結果をレポートとして出力できること

#### 1.6 GUI機能
- **FR-6.1**: ダッシュボードで予測結果を確認できること
- **FR-6.2**: チャート上にテクニカル指標と予測シグナルを表示できること
- **FR-6.3**: 銘柄選択とモデル設定をGUIから変更できること
- **FR-6.4**: ダークモードとライトモードを切り替えられること
- **FR-6.5**: バックテストレポートをGUI上で閲覧できること

### 2. 非機能要件

#### 2.1 性能要件
- **NFR-1.1**: 1銘柄の予測は10秒以内に完了すること
- **NFR-1.2**: 日経225全銘柄のデータ収集は30分以内に完了すること
- **NFR-1.3**: GUI操作に対するレスポンスは1秒以内であること
- **NFR-1.4**: モデル学習時にGPUメモリを効率的に使用すること

#### 2.2 拡張性要件（スケーラビリティ）
- **NFR-2.1**: 新しいテクニカル指標を容易に追加できること
- **NFR-2.2**: 新しいモデルアルゴリズムを容易に追加できること
- **NFR-2.3**: 銘柄数を動的に増減できること
- **NFR-2.4**: 予測期間（1週間→1ヶ月）を設定で変更できること

#### 2.3 保守性要件
- **NFR-3.1**: コードは明確な命名規則に従うこと
- **NFR-3.2**: 各モジュールは単一責任の原則に従うこと
- **NFR-3.3**: ユニットテストのカバレッジは80%以上とすること
- **NFR-3.4**: ドキュメントは最新の状態を保つこと

#### 2.4 可観測性要件（オブザーバビリティ）
- **NFR-4.1**: 全ての重要な処理をログに記録すること
- **NFR-4.2**: エラー発生時は詳細なスタックトレースを記録すること
- **NFR-4.3**: データ収集の成功/失敗を監視できること
- **NFR-4.4**: モデルのパフォーマンスを時系列で追跡できること

#### 2.5 ユーザビリティ要件
- **NFR-5.1**: 初回起動時に設定ウィザードを表示すること
- **NFR-5.2**: エラーメッセージは分かりやすい日本語で表示すること
- **NFR-5.3**: 主要な操作はショートカットキーで実行できること
- **NFR-5.4**: ツールチップで各機能の説明を表示すること

#### 2.6 互換性要件
- **NFR-6.1**: Windows 10以降で動作すること
- **NFR-6.2**: Python 3.9以降に対応すること
- **NFR-6.3**: 設定ファイル形式は後方互換性を保つこと

## 技術スタック

### プログラミング言語
- **Python 3.9+**: メイン開発言語

### GUI
- **Tkinter**: Python標準GUIライブラリ（商用利用可）
- **ttkbootstrap**: モダンなUIテーマ、ダークモード対応

### データ収集
- **yfinance**: Yahoo Finance APIラッパー（株価データ）
- **NewsAPI**: ニュース記事取得API
- **requests**: HTTPクライアント
- **schedule**: スケジューラー

### データ処理
- **pandas**: データ操作
- **numpy**: 数値計算
- **TA-Lib**: テクニカル指標計算

### 機械学習
- **scikit-learn**: 基本的な機械学習アルゴリズム
- **XGBoost**: 勾配ブースティング
- **LightGBM**: 高速勾配ブースティング
- **optuna**: ハイパーパラメータ最適化

### ディープラーニング
- **TensorFlow 2.x / Keras**: ディープラーニングフレームワーク
- **transformers**: 自然言語処理（感情分析）

### バックテスト
- **backtrader**: バックテストフレームワーク

### 可視化
- **matplotlib**: グラフ描画
- **plotly**: インタラクティブチャート
- **seaborn**: 統計可視化

### データベース
- **SQLite**: 軽量データベース
- **SQLAlchemy**: ORMライブラリ

### その他
- **PyYAML**: YAML設定ファイル
- **loguru**: 高度なログ管理
- **pytest**: テストフレームワーク
- **black**: コードフォーマッター

## 設計方針

### 1. アーキテクチャ設計原則

#### 1.1 レイヤードアーキテクチャ
- 各層は明確な責任を持ち、上位層のみが下位層に依存
- 層間のインターフェースを明確に定義し、疎結合を保つ
- 各層は独立してテスト可能

#### 1.2 モジュール設計
- 単一責任の原則（SRP）: 各クラス・関数は一つの責任のみを持つ
- オープン・クローズドの原則（OCP）: 拡張に開き、修正に閉じる
- 依存性逆転の原則（DIP）: 抽象に依存し、具象に依存しない

#### 1.3 デザインパターン
- **Factory Pattern**: モデル生成に使用
- **Strategy Pattern**: 異なる取引戦略の実装
- **Observer Pattern**: データ更新通知
- **Singleton Pattern**: 設定管理、ログ管理

### 2. データ設計

#### 2.1 データベーススキーマ
```sql
-- 株価データ
CREATE TABLE stock_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    date DATE NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date)
);

-- ニュースデータ
CREATE TABLE news_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    title TEXT NOT NULL,
    content TEXT,
    url TEXT,
    published_at TIMESTAMP,
    sentiment_score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 予測結果
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    prediction_date DATE NOT NULL,
    target_date DATE NOT NULL,
    predicted_price REAL,
    confidence_lower REAL,
    confidence_upper REAL,
    model_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- モデル情報
CREATE TABLE models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    model_type TEXT,
    parameters TEXT,
    performance_metrics TEXT,
    file_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 2.2 データフロー
```
[外部API] → [Collector] → [Raw Data] → [Processor] → [Features] → [Model] → [Predictions]
                ↓                          ↓                         ↓
            [Database]                  [Database]               [Database]
```

### 3. モデル設計

#### 3.1 モデル階層
```
BaseModel (抽象基底クラス)
    ├── MLModel (機械学習モデル)
    │   ├── XGBoostModel
    │   ├── LightGBMModel
    │   └── RandomForestModel
    ├── DeepLearningModel (ディープラーニングモデル)
    │   ├── LSTMModel
    │   ├── GRUModel
    │   └── TransformerModel
    └── EnsembleModel (アンサンブルモデル)
        ├── StackingModel
        └── VotingModel
```

#### 3.2 特徴量設計
- **価格特徴量**: OHLCV、リターン、ボラティリティ
- **テクニカル特徴量**: SMA、EMA、RSI、MACD、Bollinger Bands等
- **ラグ特徴量**: 過去N日の値
- **ローリング統計**: 移動平均、移動標準偏差
- **感情特徴量**: ニュース感情スコア、感情変化率
- **市場特徴量**: 日経平均、TOPIX等のマクロ指標

#### 3.3 予測ターゲット
- **主ターゲット**: 1週間後の終値
- **補助ターゲット**:
  - 価格変化率（%）
  - 上昇/下降/横ばいの分類
  - 最高値・最安値

### 4. GUI設計

#### 4.1 画面構成
```
Main Window
├── Menu Bar
│   ├── File (設定保存/読込、終了)
│   ├── Data (データ更新、履歴表示)
│   ├── Model (モデル訓練、評価)
│   ├── Backtest (バックテスト実行、レポート)
│   └── Help (ヘルプ、バージョン情報)
├── Tool Bar (よく使う機能のクイックアクセス)
├── Side Panel (銘柄リスト、フィルター)
└── Main Area (タブ切り替え)
    ├── Dashboard Tab (概要、アラート)
    ├── Chart Tab (チャート、インジケーター)
    ├── Prediction Tab (予測結果、信頼区間)
    ├── Backtest Tab (バックテスト結果)
    └── Settings Tab (各種設定)
```

#### 4.2 推奨GUI機能
- **ウォッチリスト**: お気に入り銘柄の登録
- **アラート機能**: 予測シグナル発生時の通知
- **比較機能**: 複数銘柄の比較表示
- **エクスポート機能**: レポートのPDF/Excel出力
- **履歴表示**: 過去の予測精度の推移
- **リアルタイム更新**: データ収集中の進捗表示

### 5. エラーハンドリング

#### 5.1 エラー分類
- **データエラー**: API接続失敗、データ欠損
- **モデルエラー**: 学習失敗、予測失敗
- **システムエラー**: メモリ不足、ディスク容量不足
- **ユーザーエラー**: 不正な入力、設定ミス

#### 5.2 エラー処理方針
- 全てのエラーをログに記録
- ユーザーには分かりやすいエラーメッセージを表示
- 可能な場合は自動リトライ（最大3回）
- クリティカルエラー時は安全にシャットダウン

### 6. ログ設計

#### 6.1 ログレベル
- **DEBUG**: 詳細なデバッグ情報
- **INFO**: 通常の動作情報
- **WARNING**: 警告（処理は継続）
- **ERROR**: エラー（処理失敗）
- **CRITICAL**: クリティカルエラー（システム停止）

#### 6.2 ログフォーマット
```
[2024-01-14 10:30:15] [INFO] [stock_collector] Fetching data for 7203.T
[2024-01-14 10:30:16] [ERROR] [news_collector] API rate limit exceeded, retrying in 60s
```

### 7. テスト戦略

#### 7.1 テスト種類
- **ユニットテスト**: 各関数・クラスの単体テスト
- **統合テスト**: モジュール間の連携テスト
- **E2Eテスト**: システム全体の動作テスト

#### 7.2 テストカバレッジ目標
- 全体: 80%以上
- コア機能: 90%以上

### 8. パフォーマンス最適化

#### 8.1 データ収集の最適化
- 並列処理による高速化
- キャッシングによる重複取得の回避
- レート制限への対応

#### 8.2 モデル学習の最適化
- GPU活用（TensorFlow GPU版）
- バッチ処理による効率化
- 早期停止（Early Stopping）

#### 8.3 GUI表示の最適化
- 遅延ローディング
- 非同期データ取得
- キャンバスの効率的な描画

### 9. セキュリティ

#### 9.1 APIキー管理
- 設定ファイル（api_keys.yaml）に保存
- .gitignoreに追加（バージョン管理から除外）
- 環境変数からの読み込みもサポート

#### 9.2 データ保護
- データベースファイルのアクセス権限制限
- 個人情報は保存しない

## データソース

### 株価データ
- **提供元**: Yahoo Finance
- **ライブラリ**: yfinance
- **対象**: 日経225銘柄
- **期間**: 2014年1月〜現在
- **頻度**: 日次、分足（必要に応じて）
- **取得データ**: 始値、高値、安値、終値、出来高

### ニュースデータ
- **提供元**: NewsAPI
- **プラン**: 無料プラン（1日100リクエスト）
- **言語**: 日本語
- **対象**: 企業関連ニュース、市場ニュース
- **取得データ**: タイトル、本文、公開日時、URL

## インストール方法（予定）

```bash
# リポジトリのクローン
git clone https://github.com/yourusername/JTrading_System.git
cd JTrading_System

# 仮想環境の作成
python -m venv venv
venv\Scripts\activate  # Windows

# 依存パッケージのインストール
pip install -r requirements.txt

# 設定ファイルの作成
copy config\config.yaml.example config\config.yaml
# config.yamlを編集してAPIキーを設定

# データベースの初期化
python setup.py init_db

# アプリケーションの起動
run.bat  # または python src/main.py
```

## 使用方法（予定）

### 基本的な流れ
1. **初回設定**: APIキーの設定、銘柄リストの選択
2. **データ収集**: 過去データの一括取得
3. **モデル訓練**: 機械学習モデルの訓練
4. **バックテスト**: 過去データでの戦略検証
5. **予測実行**: 最新データでの予測
6. **結果確認**: ダッシュボードでの確認

## 開発ロードマップ

### Phase 1: コア機能実装（MVP）
- [ ] データ収集モジュール
- [ ] テクニカル指標計算
- [ ] 基本的な機械学習モデル（XGBoost）
- [ ] シンプルなバックテストエンジン
- [ ] 基本的なGUI

### Phase 2: 機能拡張
- [ ] ニュース感情分析
- [ ] ディープラーニングモデル（LSTM）
- [ ] 高度なバックテスト機能
- [ ] GUI機能の充実

### Phase 3: 最適化・改善
- [ ] パフォーマンス最適化
- [ ] AutoML対応
- [ ] より高度な特徴量エンジニアリング
- [ ] レポート機能の強化

### Phase 4: 運用改善
- [ ] リアルタイム予測機能
- [ ] アラート・通知機能
- [ ] クラウド連携
- [ ] モバイル対応

## ライセンス

TBD

## 開発者

TBD

## 免責事項

本システムは投資判断の補助を目的としたツールです。本システムの予測結果に基づく投資の成果について、開発者は一切の責任を負いません。投資は自己責任で行ってください。

## 参考文献

- [Yahoo Finance API Documentation](https://finance.yahoo.com/)
- [NewsAPI Documentation](https://newsapi.org/docs)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [backtrader Documentation](https://www.backtrader.com/)

---

**最終更新**: 2026-01-14
**バージョン**: 0.1.0 (要件定義・設計フェーズ)
