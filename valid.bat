@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================================
echo     АВТОМАТИЗИРОВАННАЯ ОБРАБОТКА ЭКГ ДАННЫХ
echo ============================================================
echo.

:: ============================================================
:: ШАГ 1: ЗАПУСК CSV2DART.PY
:: ============================================================
echo [1/3] ЗАПУСК КОНВЕРТЕРА CSV -> DART
echo ------------------------------------------------------------
echo.

python csv2dart.py

if errorlevel 1 (
    echo.
    echo ОШИБКА: Не удалось выполнить csv2dart.py
    echo Проверьте наличие файла и корректность данных
    pause
    exit /b 1
)

echo.
echo Конвертация завершена успешно!
echo.

:: ============================================================
:: ШАГ 2: ОБРАБОТКА ВСЕХ .DART ФАЙЛОВ
:: ============================================================
echo [2/3] ЗАПУСК PAN-TOMPKINS ДЛЯ ВСЕХ .DART ФАЙЛОВ
echo ------------------------------------------------------------

:: Поиск всех подпапок с .dart файлами
set "DART_ROOT=DATA\DART-DATA"
set "OUTPUT_ROOT=DETECTED-PEAKS"

if not exist "%DART_ROOT%" (
    echo ОШИБКА: Папка %DART_ROOT% не найдена!
    echo Убедитесь, что csv2dart.py создал файлы в этой папке
    pause
    exit /b 1
)

:: Создаем корневую папку для выходных данных
if not exist "%OUTPUT_ROOT%" mkdir "%OUTPUT_ROOT%"

:: Счетчики
set total_files=0
set processed_files=0
set failed_files=0

:: Рекурсивный обход всех подпапок в DART-DATA
for /d %%d in ("%DART_ROOT%\*") do (
    :: Определяем имя базы данных (имя подпапки)
    set "db_name=%%~nxd"
    set "output_dir=%OUTPUT_ROOT%\!db_name!-PEAKS"
    
    :: Создаем выходную папку для этой базы
    if not exist "!output_dir!" mkdir "!output_dir!"
    
    echo.
    echo Обработка базы данных: !db_name!
    echo Выходная папка: !output_dir!
    
    :: Обработка всех .dart файлов в подпапке
    for %%f in ("%%d\*.dart") do (
        set /a total_files+=1
        set "input_file=%%f"
        set "output_file=!output_dir!\%%~nxf"
        set "output_file=!output_file:.dart=_peaks.txt!"
        
        echo   - Обработка: %%~nxf ...
        
        :: Запуск pan-tompkins-alg.dart
        dart run pan-tompkins-alg.dart "!input_file!" "!output_dir!" >nul 2>&1
        
        if errorlevel 1 (
            set /a failed_files+=1
            echo     [ОШИБКА] Файл %%~nxf не обработан
        ) else (
            set /a processed_files+=1
            echo     [ГОТОВО] %%~nxf
        )
    )
)

echo.
echo ------------------------------------------------------------
echo Статистика обработки:
echo   - Всего файлов: %total_files%
echo   - Успешно: %processed_files%
echo   - С ошибками: %failed_files%
echo.

if %failed_files% gtr 0 (
    echo ВНИМАНИЕ: Некоторые файлы не были обработаны
    echo Проверьте логи ошибок выше
)

:: ============================================================
:: ШАГ 3: ЗАПУСК WFDB ДЛЯ ВАЛИДАЦИИ
:: ============================================================
echo.
echo [3/3] ЗАПУСК ВАЛИДАЦИИ WFDB
echo ------------------------------------------------------------

set "ANNOTATIONS_ROOT=DATA\DATABASES"
set "RESULTS_ROOT=RESULTS"
set validation_count=0

if not exist "%RESULTS_ROOT%" mkdir "%RESULTS_ROOT%"

:: Для каждой базы данных
for /d %%d in ("%ANNOTATIONS_ROOT%\*") do (
    set "db_name=%%~nxd"
    set "peaks_dir=%OUTPUT_ROOT%\!db_name!-PEAKS"
    set "results_dir=%RESULTS_ROOT%\!db_name!-RESULTS"
    
    :: Проверяем наличие папки с пиками
    if exist "!peaks_dir!" (
        if not exist "!results_dir!" mkdir "!results_dir!"
        
        echo.
        echo Валидация базы: !db_name!
        
        :: Для каждого файла с пиками ищем соответствующие аннотации
        for %%p in ("!peaks_dir!\*_peaks.txt") do (
            set "base_name=%%~np"
            set "base_name=!base_name:_peaks=!"
            
            :: Ищем файл аннотации в папке базы данных
            set "annotation_file=%%d\!base_name!annotations.txt"
            
            if exist "!annotation_file!" (
                set /a validation_count+=1
                set "result_file=!results_dir!\!base_name!_validation.txt"
                
                echo   - Валидация: !base_name!
                
                :: Запуск WFDB.py
                python WFDB.py "!annotation_file!" "%%p" "!result_file!"
                
                if errorlevel 1 (
                    echo     [ОШИБКА] Валидация !base_name! не выполнена
                ) else (
                    echo     [ГОТОВО] !base_name!
                )
            ) else (
                echo   - [ПРОПУСК] Аннотации не найдены: !base_name!annotations.txt
            )
        )
    )
)

echo.
echo ------------------------------------------------------------
echo ВСЕ ОПЕРАЦИИ ЗАВЕРШЕНЫ!
echo ------------------------------------------------------------
echo.
echo Итоговая статистика:
echo   - Обработано .dart файлов: %processed_files%
echo   - Выполнено валидаций: %validation_count%
echo.
echo Результаты сохранены в:
echo   - Пики: %OUTPUT_ROOT%
echo   - Валидация: %RESULTS_ROOT%
echo.
pause