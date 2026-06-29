import csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import multiprocessing
import numpy as np
import sys

# Доступные базы данных
AVAILABLE_DATABASES = ["MIH-BIN", "AHA", "NSTDB"]

def convert_csv_to_dart(csv_filename, dart_filename, sampling_freq):
    """
    Конвертирует CSV файл с данными ECG в формат, аналогичный data.dart
    БЕЗ НОРМАЛИЗАЦИИ - сохраняет исходные значения MLII как double
    """
    csv_path = Path(csv_filename)
    if not csv_path.exists():
        raise FileNotFoundError(f"Файл {csv_filename} не найден")

    data_mlII = []
    
    try:
        data = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=1, dtype=np.float64)
        data_mlII = data.tolist()
    except Exception:
        data_mlII = []
    
    if not data_mlII:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            try:
                next(reader)
            except StopIteration:
                raise ValueError(f"CSV файл {csv_filename} пуст")

            for row in reader:
                if len(row) >= 3:
                    try:
                        data_mlII.append(float(row[1]))
                    except ValueError:
                        continue

    if not data_mlII:
        raise ValueError(f"Не удалось прочитать данные из CSV файла {csv_filename}")

    chunks = []
    chunk_size = 6
    for i in range(0, len(data_mlII), chunk_size):
        chunk = data_mlII[i:i + chunk_size]
        chunks.append("  " + ", ".join(f"{x:.1f}" for x in chunk))
    
    chunks_with_comma = [chunk + "," if i < len(chunks) - 1 else chunk for i, chunk in enumerate(chunks)]
    
    dart_content = f"""// ECG Data from {csv_path.name}
// Sampling Freq {sampling_freq}
// Raw MLII values (no normalization)

int samplingFreq = {sampling_freq};
List<double> data = [
{chr(10).join(chunks_with_comma)}
];"""

    output_dir = Path(dart_filename).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(dart_filename, 'w', encoding='utf-8') as f:
        f.write(dart_content)

    return len(data_mlII)

def process_file_wrapper(args):
    """Обёртка для использования с ProcessPoolExecutor"""
    csv_file, output_file, sampling_freq = args
    try:
        num_points = convert_csv_to_dart(
            csv_filename=csv_file,
            dart_filename=output_file,
            sampling_freq=sampling_freq
        )
        return (Path(csv_file).name, True, num_points, None)
    except Exception as e:
        return (Path(csv_file).name, False, 0, str(e))

def main():
    """Основная функция"""
    # Проверка аргументов командной строки
    if len(sys.argv) < 2:
        print("Ошибка: не указано название базы данных")
        print(f"Использование: python csv2dart.py <название_базы>")
        print(f"Доступные базы: {', '.join(AVAILABLE_DATABASES)}")
        return
    
    database = sys.argv[1].upper()
    
    # Проверка, существует ли указанная база данных
    if database not in [db.upper() for db in AVAILABLE_DATABASES]:
        print(f"Ошибка: база данных '{database}' не найдена")
        print(f"Доступные базы: {', '.join(AVAILABLE_DATABASES)}")
        return
    
    # Находим правильное название базы (с правильным регистром)
    for db in AVAILABLE_DATABASES:
        if db.upper() == database:
            database = db
            break
    
    script_dir = Path(__file__).parent
    
    # Формируем пути на основе выбранной базы
    input_dir = script_dir / "DATA" / "DATABASES" / database
    output_dir = script_dir / "DATA" / "DART-DATA" / f"{database}-DART"
    
    if not input_dir.exists():
        print(f"Ошибка: Папка {input_dir} не найдена")
        print(f"Убедитесь, что база данных '{database}' существует в папке DATA/DATABASES/")
        return
    
    # Запрашиваем частоту дискретизации
    while True:
        freq_input = input("\nВведите частоту дискретизации (Гц): ").strip()
        try:
            sampling_freq = int(freq_input)
            if sampling_freq <= 0:
                print("Частота должна быть положительным числом")
                continue
            break
        except ValueError:
            print("Пожалуйста, введите целое число")
    
    # Создаем выходную папку
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Находим все CSV файлы
    csv_files = list(input_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"В папке {input_dir} не найдено CSV файлов")
        return
    
    total_files = len(csv_files)
    
    # Определяем количество процессов
    cpu_count = multiprocessing.cpu_count()
    max_workers = min(cpu_count * 2, total_files)
    
    start_time = time.time()
    
    process_args = []
    for csv_file in csv_files:
        output_file = output_dir / csv_file.name
        output_file = output_file.with_suffix(".dart")
        process_args.append((str(csv_file), str(output_file), sampling_freq))
    
    results = []
    completed = 0
    successful = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file_wrapper, args) for args in process_args]
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            
            if result[1]:  # успешно
                successful += 1
            else:
                failed += 1
            
            elapsed = time.time() - start_time
            if completed > 0:
                avg_time = elapsed / completed
                remaining = (total_files - completed) * avg_time
                print(f"\rКонвертация данных: обработано {completed} из {total_files} файлов, "
                      f"успешно: {successful}, ошибок: {failed}, осталось ~{remaining:.1f} сек", end="")
    
    print()
    print()
    
    # Вывод статистики ошибок
    if failed > 0:
        print("Ошибки при обработке файлов:")
        for filename, success, _, error in results:
            if not success:
                print(f"  - {filename}: {error}")
    
    total_time = time.time() - start_time

if __name__ == "__main__":
    main()