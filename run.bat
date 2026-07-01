@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 
title Выбор базы данных

:menu
cls
echo =============================================================
echo                  Выберите базу данных:
echo =============================================================
echo                  1 - MIT-BIH
echo                  2 - AHA
echo                  3 - NSTDB
set /p choice="Введите номер (1-3): "
echo.

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
set dart_dir=%project_root%DATA\CSV-DATA\%db_name%-CSV
set peaks_dir=%project_root%DATA\DETECTED-PEAKS\%db_name%-PEAKS
set res_dir=%project_root%DATA\RESULTS\%db_name%-RESULTS
set ann_dir=%project_root%DATA\DATABASES\%db_name%

echo Запущена конвертация файлов данных...
python %project_root%sys\wfdb2csv.py "%input_dir%" "%dart_dir%"
echo.
echo Конвертация завершена успешно!
echo.
echo Запущена обработка данных валидируемым алгоритмом...
echo.

for %%f in ("%dart_dir%\*.csv") do (
    echo =============================================================
    echo Обработка: %%~nxf
    
    set "skip=0"
    
    if "%db_name%"=="MIT-BIH" (
        for %%n in (102 104 107 217) do (
            if "%%~nf"=="%%n" set "skip=1"
        )
    )

     if "%db_name%"=="NSTDB" (
        for %%n in (bw em ma) do (
            if "%%~nf"=="%%n" set "skip=1"
        )
    )
    
    if !skip!==0 (
        :: Формируем путь к .atr файлу в peaks_dir
        set "atr_file=%peaks_dir%\%%~nf.atr"
        
        :: Запускаем pan-tompkins-alg и передаем его вывод в list2atr.py
        cmd /c dart run %project_root%algs\pan-tompkins-alg.dart "%%f" | python %project_root%sys\list2atr.py "!atr_file!"
        
        if errorlevel 1 (
            echo Ошибка: Обработка файла %%~nxf
        ) else (
            echo Файл %%~nf успешно обработан
        )
    ) else (
        echo Файл %%~nf пропущен, эта запись не используется при валидации
    )
)
echo.
echo Запущена валидация...
python %project_root%sys\validation.py "%input_dir%" "%peaks_dir%" "%res_dir%" --summary
echo.
echo Валидация успешно завершена!
echo.
echo Результаты работы программы сохранены в %res_dir%
pause