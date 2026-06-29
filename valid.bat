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
echo.
python csv2dart.py "%db_name%"
echo.

:: Проверяем, существует ли папка с DART-данными
set input_dir=DATA\DART-DATA\%db_name%-DART
if not exist "%input_dir%" (
    echo Ошибка: Папка %input_dir% не найдена!
    pause
    goto menu
)

:: Создаём выходную папку, если её нет
set output_dir=DATA\DETECTED-PEAKS\%db_name%-PEAKS
if not exist "%output_dir%" mkdir "%output_dir%"


:: Обрабатываем каждый Dart-файл в папке
for %%f in ("%input_dir%\*.dart") do @(
    echo.
    echo Обработка: %%~nxf
    dart run pan-tompkins-alg.dart "%%f" "%output_dir%"
    if errorlevel 1 (
        echo Ошибка: файл %%~nxf не обработан
    ) else (
        echo Успешно: файл %%~nxf обработан
    )
)

echo.
echo ==============================
echo Все файлы обработаны!
echo Результаты в: %output_dir%
echo ==============================
pause