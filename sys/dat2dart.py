import wfdb
import argparse
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def convert_dat_to_dart(record_name, dart_filename, physical=True):
    """
    Конвертирует WFDB запись (.dat + .hea) в Dart формат.
    По умолчанию сохраняет физические значения (например, мВ) как double.
    """
    hea_path = Path(record_name).with_suffix('.hea')
    if not hea_path.exists():
        raise FileNotFoundError(f"Файл заголовка {hea_path} не найден")

    try:
        # Читаем запись. physical=True возвращает значения в мВ и т.д. [citation:6]
        record = wfdb.rdrecord(str(record_name), physical=physical)
    except Exception as e:
        raise RuntimeError(f"Ошибка чтения WFDB записи: {e}")

    if record.p_signal is None or record.p_signal.size == 0:
        raise ValueError("Не удалось прочитать сигналы из записи")

    # Для простоты берем первый канал (индекс 0). Можно расширить для многоканальных.
    data = record.p_signal[:, 0].flatten().tolist()

    # Форматирование в стиле вашего csv2dart.py
    chunk_size = 6
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        # Формат с одним знаком после запятой
        chunks.append("  " + ", ".join(f"{x:.1f}" for x in chunk))

    chunks_with_comma = [chunk + "," if i < len(chunks) - 1 else chunk for i, chunk in enumerate(chunks)]

    dart_content = f"""// ECG Data from {Path(record_name).name}
// Sampling Freq {record.fs}
// Physical units (e.g., mV)
// Channel: {record.sig_name[0] if record.sig_name else 'Channel 0'}

int samplingFreq = {record.fs};
List<double> data = [
{chr(10).join(chunks_with_comma)}
];"""

    output_dir = Path(dart_filename).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(dart_filename, 'w', encoding='utf-8') as f:
        f.write(dart_content)

    return len(data)

def process_file_wrapper(args):
    """Обёртка для многопроцессорной обработки."""
    record_path, output_path = args
    try:
        num_points = convert_dat_to_dart(
            record_name=str(record_path), # Передаем базовое имя без расширения
            dart_filename=str(output_path)
        )
        return (Path(record_path).name, True, num_points, None)
    except Exception as e:
        return (Path(record_path).name, False, 0, str(e))

def main():
    parser = argparse.ArgumentParser(description='Конвертирует WFDB файлы (.dat/.hea) в Dart формат')
    parser.add_argument('input_dir', help='Путь к директории с файлами WFDB записей')
    parser.add_argument('output_dir', help='Путь к директории для сохранения .dart файлов')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Количество параллельных процессов (по умолчанию: число ядер CPU)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Ошибка: Папка {input_dir} не найдена или не является директорией")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Находим все .dat файлы, чтобы получить базовые имена записей
    dat_files = list(input_dir.glob("*.dat"))
    if not dat_files:
        print(f"В папке {input_dir} не найдено .dat файлов")
        return

    # Для каждой записи нужен соответствующий .hea файл
    process_args = []
    for dat_path in dat_files:
        record_base = dat_path.stem
        hea_path = input_dir / f"{record_base}.hea"
        if hea_path.exists():
            output_path = output_dir / f"{record_base}.dart"
            # Передаем путь к записи (без расширения)
            process_args.append((input_dir / record_base, output_path))

    if not process_args:
        print("Не найдено пар (.dat/.hea) для конвертации")
        return

    total_files = len(process_args)
    cpu_count = multiprocessing.cpu_count()
    max_workers = min(args.workers if args.workers else cpu_count * 2, total_files)

    start_time = time.time()
    successful, failed = 0, 0
    error_messages = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file_wrapper, args) for args in process_args]

        for future in as_completed(futures):
            result = future.result()
            if result[1]:
                successful += 1
            else:
                failed += 1
                if result[3]:
                    error_messages.append(f"  - {result[0]}: {result[3]}")

            completed = successful + failed
            elapsed = time.time() - start_time
            avg_time = elapsed / completed if completed > 0 else 0
            remaining = (total_files - completed) * avg_time
            print(f"\rКонвертация: {completed}/{total_files} | OK: {successful} | ERR: {failed} | ~{remaining:.0f}с", end="")

    print()
    if failed > 0:
        print("\nОшибки:")
        for error in error_messages:
            print(error)

if __name__ == "__main__":
    main()