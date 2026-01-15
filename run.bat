@echo off
setlocal enabledelayedexpansion
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
echo   1. 拡張GUI モード（チャート表示対応）★推奨
echo   2. シンプルGUI モード
echo   3. データ収集のみ
echo   4. モデル訓練のみ
echo   5. バックテストのみ
echo   6. フルワークフロー
echo.

set /p mode="モード番号を入力 (1-6, Enter=1): "

REM デフォルト値の設定
if "!mode!"=="" set mode=1

REM モードに応じて処理を分岐
if "!mode!"=="1" goto MODE_GUI2
if "!mode!"=="2" goto MODE_GUI
if "!mode!"=="3" goto MODE_COLLECT
if "!mode!"=="4" goto MODE_TRAIN
if "!mode!"=="5" goto MODE_BACKTEST
if "!mode!"=="6" goto MODE_FULL

REM 無効なモード番号
echo [エラー] 無効なモード番号です
pause
exit /b 1

:MODE_GUI2
echo.
echo 拡張GUIモードで起動中（チャート表示対応）...
python src\main.py --mode gui2
goto END

:MODE_GUI
echo.
echo シンプルGUIモードで起動中...
python src\main.py --mode gui
goto END

:MODE_COLLECT
echo.
echo データ収集を開始中...
python src\main.py --mode collect
goto END

:MODE_TRAIN
echo.
set /p symbol="銘柄コードを入力 (例: 7203.T, Enter=7203.T): "
if "!symbol!"=="" set symbol=7203.T
echo モデル訓練を開始中 (銘柄: !symbol!)...
python src\main.py --mode train --symbol !symbol!
goto END

:MODE_BACKTEST
echo.
set /p symbol="銘柄コードを入力 (例: 7203.T, Enter=7203.T): "
if "!symbol!"=="" set symbol=7203.T
echo バックテストを開始中 (銘柄: !symbol!)...
python src\main.py --mode backtest --symbol !symbol!
goto END

:MODE_FULL
echo.
set /p symbol="銘柄コードを入力 (例: 7203.T, Enter=7203.T): "
if "!symbol!"=="" set symbol=7203.T
echo フルワークフローを開始中 (銘柄: !symbol!)...
python src\main.py --mode full --symbol !symbol!
goto END

:END
echo.
echo ================================================
echo 処理が完了しました
echo ================================================
echo.
pause
