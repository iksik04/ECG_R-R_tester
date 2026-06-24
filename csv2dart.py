import csv
import re
import os
from pathlib import Path

def parse_timestamp(timestamp_str):
    """
    Парсит временную метку вида "0:00.050" или "0:00.214" в секунды
    """
    parts = timestamp_str.strip().split(':')
    if len(parts) == 2:
        minutes = float(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    else:
        return float(timestamp_str)

def extract_timestamp_from_line(line):
    """
    Извлекает временную метку из строки вида:
    "    0:00.050       18     +    0    0    0	(N"
    """
    match = re.search(r'(\d+):(\d+\.\d+)', line)
    if match:
        minutes = int(match.group(1))
        seconds = float(match.group(2))
        return minutes * 60 + seconds
    return None

def detect_sampling_freq_from_annotations(annotations_filename, csv_filename):
    """
    Определяет частоту дискретизации на основе временных меток из файла аннотаций
    """
    timestamps = []
    
    try:
        with open(annotations_filename, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and not line.startswith('Time'):
                    ts = extract_timestamp_from_line(line)
                    if ts is not None:
                        timestamps.append(ts)
    except FileNotFoundError:
        return None
    
    if len(timestamps) < 2:
        return None
    
    intervals = []
    for i in range(1, len(timestamps)):
        interval = timestamps[i] - timestamps[i-1]
        if interval > 0 and interval < 10:
            intervals.append(interval)
    
    if not intervals:
        return None
    
    avg_interval = sum(intervals) / len(intervals)
    
    sample_numbers = []
    with open(csv_filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return None
        for row in reader:
            if len(row) >= 1:
                try:
                    sample_num = int(row[0].strip("'"))
                    sample_numbers.append(sample_num)
                except ValueError:
                    pass
    
    if not sample_numbers:
        return None
    
    annotation_samples = []
    with open(annotations_filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('Time'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        sample_num = int(parts[1])
                        annotation_samples.append(sample_num)
                    except ValueError:
                        pass
    
    if len(annotation_samples) < 2:
        return None
    
    n_samples = []
    with open(annotations_filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('Time'):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        sample_num = int(parts[1])
                        beat_type = parts[2]
                        if beat_type == 'N':
                            n_samples.append(sample_num)
                    except (ValueError, IndexError):
                        pass
    
    if len(n_samples) < 2:
        n_samples = annotation_samples[:50]
    
    if len(n_samples) < 2:
        return None
    
    n_samples = n_samples[:50] if len(n_samples) > 50 else n_samples
    
    n_timestamps = []
    with open(annotations_filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('Time'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        sample_num = int(parts[1])
                        if sample_num in n_samples:
                            ts = extract_timestamp_from_line(line)
                            if ts is not None:
                                n_timestamps.append((sample_num, ts))
                    except ValueError:
                        pass
    
    if len(n_timestamps) < 2:
        return None
    
    n_timestamps.sort(key=lambda x: x[0])
    
    time_intervals = []
    sample_intervals = []
    for i in range(1, len(n_timestamps)):
        sample_diff = n_timestamps[i][0] - n_timestamps[i-1][0]
        time_diff = n_timestamps[i][1] - n_timestamps[i-1][1]
        if sample_diff > 0 and time_diff > 0:
            sample_intervals.append(sample_diff)
            time_intervals.append(time_diff)
    
    if not sample_intervals:
        return None
    
    avg_sample_interval = sum(sample_intervals) / len(sample_intervals)
    avg_time_interval = sum(time_intervals) / len(time_intervals)
    
    if avg_time_interval > 0:
        sampling_freq = avg_sample_interval / avg_time_interval
        return round(sampling_freq)
    
    return None

def convert_csv_to_dart(csv_filename, dart_filename, sampling_freq=None):
    """
    Конвертирует CSV файл с данными ECG в формат, аналогичный data.dart
    
    Ожидается, что CSV файл имеет столбцы: 'sample #', 'MLII', 'V5'
    Используются данные из столбца MLII.
    
    Аргументы:
        csv_filename: путь к CSV файлу
        dart_filename: путь для сохранения Dart файла
        sampling_freq: частота дискретизации (если None, попытается определить из аннотаций)
    """
    
    csv_path = Path(csv_filename)
    if not csv_path.exists():
        raise FileNotFoundError(f"Файл {csv_filename} не найден")

    data_mlII = []

    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"CSV файл {csv_filename} пуст")

        for row in reader:
            if len(row) >= 3:
                try:
                    mlII = int(row[1])
                    data_mlII.append(mlII)
                except ValueError as e:
                    print(f"Ошибка при парсинге строки {row} в файле {csv_filename}: {e}")
                    continue

    if not data_mlII:
        raise ValueError(f"Не удалось прочитать данные из CSV файла {csv_filename}")

    # Определяем частоту дискретизации
    if sampling_freq is None:
        annotations_file = csv_path.with_name(csv_path.stem + "annotations.txt")
        if annotations_file.exists():
            detected_freq = detect_sampling_freq_from_annotations(str(annotations_file), csv_filename)
            if detected_freq is not None:
                sampling_freq = detected_freq
            else:
                sampling_freq = 125
        else:
            sampling_freq = 125

    # Нормализация данных
    sorted_data = sorted(data_mlII)
    min_val = sorted_data[0]
    max_val = sorted_data[-1]
    range_val = max_val - min_val

    if range_val == 0:
        range_val = 1

    normalized_data = []
    for val in data_mlII:
        norm = (val - min_val) / range_val
        scaled = 1.0 + norm * 1.5
        normalized_data.append(round(scaled, 4))

    # Формируем Dart файл
    dart_lines = [
        f"// ECG Data from {csv_path.name}",
        f"// Sampling Freq {sampling_freq}",
        "",
        f"int samplingFreq = {sampling_freq};",
        "List<double> data = ["
    ]

    chunk_size = 6
    for i in range(0, len(normalized_data), chunk_size):
        chunk = normalized_data[i:i + chunk_size]
        line = "  " + ", ".join(f"{x:.4f}" for x in chunk)
        if i + chunk_size < len(normalized_data):
            line += ","
        dart_lines.append(line)

    dart_lines.append("];")

    # Создаем директорию для выходных файлов если её нет
    output_dir = Path(dart_filename).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(dart_filename, 'w', encoding='utf-8') as f:
        f.write("\n".join(dart_lines))

    return len(normalized_data), sampling_freq

def main():
    """
    Конвертирует все CSV файлы из папки MIH-BIN в Dart файлы в папку MIH-BIN-DART
    """
    script_dir = Path(__file__).parent
    
    input_dir = script_dir / "MIH-BIN"
    output_dir = script_dir / "MIH-BIN-DART"
    
    print("=" * 60)
    print("ECG CSV to Dart Converter - Batch Mode")
    print("=" * 60)
    print()
    
    if not input_dir.exists():
        print(f"Ошибка: Папка {input_dir} не найдена")
        print(f"Создайте папку MIH-BIN в директории скрипта и поместите туда CSV файлы")
        return
    
    # Создаем выходную папку
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Находим все CSV файлы
    csv_files = list(input_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"В папке {input_dir} не найдено CSV файлов")
        return
    
    print(f"Найдено CSV файлов: {len(csv_files)}")
    print(f"Выходная папка: {output_dir}")
    print()
    
    total_files = 0
    total_points = 0
    
    for csv_file in sorted(csv_files):
        try:
            print(f"Обработка: {csv_file.name}...", end=" ")
            
            output_file = output_dir / csv_file.name
            output_file = output_file.with_suffix(".dart")
            
            num_points, freq = convert_csv_to_dart(
                csv_filename=str(csv_file),
                dart_filename=str(output_file)
            )
            
            total_files += 1
            total_points += num_points
            
            print(f"OK (точек: {num_points}, частота: {freq} Hz)")
            
        except Exception as e:
            print(f"ОШИБКА: {e}")
    
    print()
    print("=" * 60)
    print(f"Конвертация завершена")
    print(f"Обработано файлов: {total_files}")
    print(f"Всего точек: {total_points}")
    print(f"Выходная папка: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()