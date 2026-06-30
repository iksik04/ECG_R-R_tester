@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 
title Выбор базы данных

:menu
cls
echo ==============================
echo   Выберите базу данных:
echo ==============================
echo   1 - MIT-BIH
echo   2 - AHA
echo   3 - NSTDB
echo ==============================
set /p choice="Введите номер (1-3): "

if "%choice%"=="1" (
    set db_name=MIT-BIH
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

python %project_root%sys\dat2dart.py "%input_dir%" "%dart_dir%"

for %%f in ("%dart_dir%\*.dart") do (
    echo =================
    echo Обработка: %%~nxf
    
    set "skip=0"
    
    if "%db_name%"=="MIT-BIH" (
        for %%n in (102 104 107 217) do (
            if "%%~nf"=="%%n" set "skip=1"
        )
    )
    
    if !skip!==0 (
        cmd /c dart run %project_root%algs\pan-tompkins-alg.dart "%%f" "%peaks_dir%"
        if errorlevel 1 (
            echo Ошибка: Dart для файла %%~nxf
        ) else (
            echo Файл %%~nf прошел обработку алгоритмом
        )
    ) else (
        echo Файл %%~nf пропущен, в MIT-BIH эти записи не используются при валидации
    )
)

python %project_root%sys\validation.py "%input_dir%" "%peaks_dir%" "%res_dir%"

echo.
echo Все файлы обработаны.
pause