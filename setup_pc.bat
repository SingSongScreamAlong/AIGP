@echo off
echo ============================================================
echo   AI Grand Prix - PC Setup
echo   This installs Python dependencies for the training loop
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install from https://python.org
    pause
    exit /b 1
)

:: Check ffmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo WARNING: ffmpeg not found.
    echo Install: choco install ffmpeg
    echo    or: https://ffmpeg.org/download.html
    echo.
)

:: Install dependencies
echo Installing Python packages...
pip install ultralytics opencv-python numpy torch torchvision --upgrade
echo.

:: Check GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'WARNING: No GPU detected - training will be slow')"
echo.

echo ============================================================
echo   Setup complete!
echo.
echo   To start the 24/7 loop:
echo     1. Open DCL The Game in windowed mode
echo     2. Start a free flight or race
echo     3. Run:  python run_pc_loop.py --capture-duration 120
echo.
echo   Or for just training (no capture):
echo     python run_pc_loop.py --train-only
echo ============================================================
pause
