@echo off
chcp 65001 >nul 
title Выбор базы данных

:menu
cls
echo ==============================
echo   Выберите базу данных:
echo ==============================
echo   1 - MIH-BIN
echo   2 - AHA
echo   3 - NSTDB
echo ==============================
set /p choice="Введите номер (1-3): "

if "%choice%"=="1" (
    set db_name=MIH-BIN
    goto run
)
if "%choice%"=="2" (
    set db_name=AHA
    goto run
)
if "%choice%"=="3" (
    set db_name=NSTDB
    goto run
)

echo.
echo Неверный выбор! Попробуйте снова.
timeout /t 2 >nul
goto menu

:run
set project_root=%~dp0

set input_dir=%project_root%DATA\DATABASES\%db_name%
set dart_dir=%project_root%DATA\DART-DATA\%db_name%-DART
set peaks_dir=%project_root%DATA\DETECTED-PEAKS\%db_name%-PEAKS
set res_dir=%project_root%DATA\RESULTS\%db_name%-RESULTS
set ann_dir=%project_root%DATA\DATABASES\%db_name%

python %project_root%sys\csv2dart.py "%input_dir%" "%dart_dir%" --sampling-freq 360

for %%f in ("%dart_dir%\*.dart") do (
    echo =================
    echo Обработка: %%~nxf
    cmd /c dart run %project_root%algs\pan-tompkins-alg.dart "%%f" "%peaks_dir%"
    if errorlevel 1 (
        echo Ошибка: Dart для файла %%~nxf
    ) else (
        echo Файл %%~nf прошел обработку алгоритмом
    )
    cmd /c python %project_root%sys\WFDB.py "%ann_dir%\%%~nfannotations.txt" "%peaks_dir%\%%~nf_peaks.txt" "%res_dir%\%%~nf_results.txt"
    if errorlevel 1 (
        echo Ошибка: Python для файла %%~nxf
    ) else (
        echo Валидация для файла %%~nf завершена
    )
)

echo.
echo Все файлы обработаны.
pause