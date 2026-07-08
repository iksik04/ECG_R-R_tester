import sys
import json
import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
import traceback
from datetime import datetime

# Импорт модулей из sys
from core.wfdb_reader import read_wfdb
from core.atr_writer import save_atr
from core.validator import PeakValidator
from core.progress import print_progress
from core.config_loader import load_config


def process_record(record_name, db_dir, pred_dir, dart_cmd, dart_script, 
                   json_debug_dir=None, save_json=True):
    """
    Обрабатывает одну запись:
    - читает сигнал из WFDB
    - сохраняет JSON файл с входными данными (опционально)
    - запускает Dart-процесс с JSON-данными
    - получает пики и сохраняет их в .atr
    - сохраняет JSON файл с выходными данными (опционально)
    """
    try:
        # Чтение сигнала
        signal, fs = read_wfdb(db_dir / record_name)
        if signal is None:
            return (record_name, False, "Signal is None")
        
        # Подготовка данных для Dart
        data = {
            "record_name": record_name,
            "signal": signal.tolist(),
            "fs": fs,
            "timestamp": datetime.now().isoformat()
        }
        json_input = json.dumps(data, ensure_ascii=False)
        
        # Сохранение входного JSON файла (для отладки)
        if save_json and json_debug_dir:
            json_debug_dir = Path(json_debug_dir)
            json_debug_dir.mkdir(parents=True, exist_ok=True)
            input_json_path = json_debug_dir / f"{record_name}_input.json"
            with open(input_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Запуск Dart процесса
        proc = subprocess.Popen(
            [dart_cmd, "run", dart_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True
        )
        stdout, stderr = proc.communicate(input=json_input)
        
        if proc.returncode != 0:
            return (record_name, False, f"Dart error: {stderr.strip()}")
        
        # Парсим JSON-ответ
        try:
            result = json.loads(stdout)
            peaks = result.get("peaks", [])
            # Извлекаем дополнительную информацию, если она есть
            algorithm_info = result.get("algorithm", {})
            processing_time = result.get("processing_time_ms", 0)
        except json.JSONDecodeError:
            return (record_name, False, f"Invalid JSON output: {stdout[:100]}")
        
        # Сохранение выходного JSON файла (для отладки)
        if save_json and json_debug_dir:
            output_data = {
                "record_name": record_name,
                "peaks": peaks,
                "algorithm": algorithm_info,
                "processing_time_ms": processing_time,
                "num_peaks": len(peaks),
                "timestamp": datetime.now().isoformat()
            }
            output_json_path = json_debug_dir / f"{record_name}_output.json"
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # Запись .atr
        output_path = pred_dir / f"{record_name}.atr"
        save_atr(peaks, output_path)
        
        # Формируем сообщение с дополнительной информацией
        msg = f"Success: {len(peaks)} peaks"
        if processing_time:
            msg += f", time: {processing_time}ms"
        
        return (record_name, True, msg)
    
    except Exception as e:
        return (record_name, False, f"Exception: {str(e)}\n{traceback.format_exc()}")


def main():
    # Загрузка конфигурации
    config = load_config()
    base_dir = Path(__file__).resolve().parent
    
    databases_root = base_dir / config["paths"]["databases_root"]
    pred_annotations_root = base_dir / config["paths"]["pred_annotations_root"]
    results_root = base_dir / config["paths"]["results_root"]
    
    # Выбор базы данных
    db_names = config["databases"]
    print("Доступные базы данных:")
    for i, name in enumerate(db_names, 1):
        print(f"{i}. {name}")
    choice = input("Введите номер: ")
    try:
        idx = int(choice) - 1
        db_name = db_names[idx]
    except (ValueError, IndexError):
        print("Неверный выбор")
        return
    
    db_dir = databases_root / db_name
    if not db_dir.exists():
        print(f"Папка с базой {db_dir} не найдена")
        return
    
    pred_dir = pred_annotations_root / f"{db_name}-PRED"
    results_dir = results_root / f"{db_name}-RESULTS"
    json_debug_dir = pred_dir / "json_debug"  # Папка для JSON файлов
    
    pred_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    json_debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Список записей (по файлам .hea)
    hea_files = list(db_dir.glob("*.hea"))
    record_names = [f.stem for f in hea_files]
    
    # Пропускаем файлы из конфига
    skip_list = config.get("skip_files", {}).get(db_name, [])
    record_names = [r for r in record_names if r not in skip_list]
    if not record_names:
        print("Нет записей для обработки")
        return
    
    # Параметры Dart
    dart_cmd = config["dart"]["executable"]
    dart_script = base_dir / config["dart"]["script_path"]
    if not dart_script.exists():
        print(f"Скрипт Dart не найден: {dart_script}")
        return
    
    # Параметры сохранения JSON
    save_json = True  # Можно добавить в конфиг как параметр
    print(f"Сохранение JSON файлов в: {json_debug_dir}")
    
    # Параллельная обработка
    max_workers = config.get("max_workers", multiprocessing.cpu_count())
    total = len(record_names)
    start_time = time.time()
    successful = 0
    failed = 0
    errors = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for rn in record_names:
            future = executor.submit(
                process_record, 
                rn, 
                db_dir, 
                pred_dir, 
                dart_cmd, 
                str(dart_script),
                str(json_debug_dir),
                save_json
            )
            futures[future] = rn
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            rn, success, msg = future.result()
            if success:
                successful += 1
            else:
                failed += 1
                errors.append(f"{rn}: {msg}")
            print_progress(completed, total, successful, failed, start_time)
    
    print()  # переход на новую строку
    
    if errors:
        print("\nОшибки:")
        for err in errors:
            print(f"  {err}")

    
    # Валидация
    print("\nЗапуск валидации...")
    validator = PeakValidator(str(db_dir), str(pred_dir), str(results_dir))
    
    # Передаём параметры из конфига
    allowed_symbols = config.get("annotation_symbols", ["N", "V"])
    tolerance_ms = config.get("tolerance_ms", 150.0)
    
    # Список записей, для которых есть предсказания
    records_to_validate = [r for r in record_names if (pred_dir / f"{r}.atr").exists()]
    
    if records_to_validate:
        # Запускаем пакетную валидацию
        validator.QRS_SYMBOLS = allowed_symbols
        validator.validate_batch(
            records_to_validate,
            allowed_symbols=allowed_symbols,
            tolerance_ms=tolerance_ms
        )
        # Генерируем сводный отчёт
        summary_file = validator.generate_summary_report(records_to_validate)
        print(f"Валидация завершена. Отчёт сохранён в {summary_file}")
        
        # Выводим информацию о JSON файлах
        json_files = list(json_debug_dir.glob("*.json"))
        if json_files:
            print(f"\nСохранено JSON файлов: {len(json_files)}")
            print(f"  Входные: {len(list(json_debug_dir.glob('*_input.json')))}")
            print(f"  Выходные: {len(list(json_debug_dir.glob('*_output.json')))}")
    else:
        print("Нет записей для валидации")


if __name__ == "__main__":
    main()