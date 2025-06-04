@echo off
echo Solving WaveTopo dependency issues...
echo.

echo === Step 1: Upgrading pip ===
python -m pip install --upgrade pip
echo.

echo === Step 2: Installing compatible libraries ===
pip install librosa==0.9.2 numpy==1.21.6 soundfile==0.11.0 pyworld==0.3.2
echo.

echo === Step 3: Running WaveTopo ===
echo Press any key to start the acoustic optimizer...
pause
python wavetopo.py