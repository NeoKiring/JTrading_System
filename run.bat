@echo off
chcp 65001 >nul
REM JTrading System 起動スクリプト

echo ================================================
echo JTrading System - 日本株式AI予測システム
echo ================================================
echo.

REM 仮想環境の確認
if exist venv\Scripts\activate.bat (
    echo 仮想環境をアクティベート中...
    call venv\Scripts\activate.bat
) else (
    echo [警告] 仮想環境が見つかりません
    echo 以下のコマンドで仮想環境を作成してください:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Pythonの確認
python --version >nul 2>&1
if errorlevel 1 (
    echo [エラー] Pythonが見つかりません
    pause
    exit /b 1
)

echo.
echo 実行モードを選択してください:
echo   1. GUI モード（デフォルト）
echo   2. データ収集のみ
echo   3. モデル訓練のみ
echo   4. バックテストのみ
echo   5. フルワークフロー
echo.

set /p mode="モード番号を入力 (1-5, Enter=1): "

if "%mode%"=="" set mode=1

if "%mode%"=="1" (
    echo GUIモードで起動中...
    python src\main.py --mode gui
) else if "%mode%"=="2" (
    echo データ収集を開始中...
    python src\main.py --mode collect
) else if "%mode%"=="3" (
    set /p symbol="銘柄コードを入力 (例: 7203.T): "
    if "!symbol!"=="" set symbol=7203.T
    echo モデル訓練を開始中 (銘柄: !symbol!)...
    python src\main.py --mode train --symbol !symbol!
) else if "%mode%"=="4" (
    set /p symbol="銘柄コードを入力 (例: 7203.T): "
    if "!symbol!"=="" set symbol=7203.T
    echo バックテストを開始中 (銘柄: !symbol!)...
    python src\main.py --mode backtest --symbol !symbol!
) else if "%mode%"=="5" (
    set /p symbol="銘柄コードを入力 (例: 7203.T): "
    if "!symbol!"=="" set symbol=7203.T
    echo フルワークフローを開始中 (銘柄: !symbol!)...
    python src\main.py --mode full --symbol !symbol!
) else (
    echo [エラー] 無効なモード番号です
    pause
    exit /b 1
)

echo.
echo ================================================
echo 処理が完了しました
echo ================================================

pause
