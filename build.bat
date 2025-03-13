@echo off
echo Installing PyInstaller...
pip install pyinstaller
echo.
echo Building executable...
python -m PyInstaller --onefile --add-data "pokemon_names.txt;." "pokemon finder script.py"
echo.
echo Done! Press any key to exit.
pause