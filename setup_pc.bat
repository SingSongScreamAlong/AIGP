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
pip install ultralytics opencv-python numpy torch torchvision mujoco pyyaml --upgrade
echo.

:: Check GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'WARNING: No GPU detected - training will be slow')"
echo.

:: Check MuJoCo
python -c "import mujoco; print(f'MuJoCo: {mujoco.__version__}')" 2>nul
if errorlevel 1 (
    echo WARNING: MuJoCo import failed. Will use OpenCV fallback for data generation.
)
echo.

echo ============================================================
echo   Setup complete!
echo.
echo   FULLY AUTONOMOUS (no DCL needed):
echo     python run_pc_autonomous.py
echo     python run_pc_autonomous.py --cycles 10 --images-per-cycle 1000
echo     python run_pc_autonomous.py --quick    (smoke test)
echo.
echo   WITH DCL CAPTURE (DCL must be running):
echo     python run_pc_loop.py --capture-duration 120
echo ============================================================
pause
