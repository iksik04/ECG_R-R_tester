import wfdb
import argparse
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from utils import print_progress


def convert_dat_to_csv(record_name, csv_filename):
    """
    Конвертирует WFDB запись (.dat + .hea) в CSV формат.
    Первая строка: частота дискретизации (целое число).
    Последующие строки: значения ЭКГ (физические единицы, например мВ),
    по одному числу на строку.
    """
    hea_path = Path(record_name).with_suffix('.hea')
    if not hea_path.exists():
        raise FileNotFoundError(f"Файл заголовка {hea_path} не найден")

    try:
        record = wfdb.rdrecord(str(record_name), physical=True)
    except Exception as e:
        raise RuntimeError(f"Ошибка чтения WFDB записи: {e}")

    if record.p_signal is None or record.p_signal.size == 0:
        raise ValueError("Не удалось прочитать сигналы из записи")

    # Берём первый канал (можно расширить для многоканальных)
    data = record.p_signal[:, 0].flatten().tolist()

    # Формируем содержимое CSV
    lines = []
    lines.append(str(int(record.fs)))   # первая строка – частота
    # Каждое значение с одним знаком после запятой
    for val in data:
        lines.append(f"{val:.1f}")

    # Запись в файл
    output_dir = Path(csv_filename).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_filename, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    return len(data)


def process_file_wrapper(args):
    """Обёртка для многопроцессорной обработки."""
    record_path, output_path = args
    try:
        num_points = convert_dat_to_csv(
            record_name=str(record_path),
            csv_filename=str(output_path)
        )
        return (Path(record_path).name, True, num_points, None)
    except Exception as e:
        return (Path(record_path).name, False, 0, str(e))


def main():
    parser = argparse.ArgumentParser(
        description='Конвертирует WFDB файлы (.dat/.hea) в CSV формат для Pan-Tompkins алгоритма'
    )
    parser.add_argument('input_dir', help='Путь к директории с файлами WFDB записей')
    parser.add_argument('output_dir', help='Путь к директории для сохранения .csv файлов')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Количество параллельных процессов (по умолчанию: число ядер CPU)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Ошибка: Папка {input_dir} не найдена или не является директорией")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    dat_files = list(input_dir.glob("*.dat"))
    if not dat_files:
        print(f"В папке {input_dir} не найдено .dat файлов")
        return

    process_args = []
    for dat_path in dat_files:
        record_base = dat_path.stem
        hea_path = input_dir / f"{record_base}.hea"
        if hea_path.exists():
            output_path = output_dir / f"{record_base}.csv"
            process_args.append((input_dir / record_base, output_path))

    if not process_args:
        print("Не найдено пар (.dat/.hea) для конвертации")
        return

    total_files = len(process_args)
    max_workers = min(args.workers if args.workers else multiprocessing.cpu_count(), total_files)

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
            print_progress(completed, total_files, successful, failed, start_time)

    print()  # переход на новую строку

    if failed > 0:
        print("\nОшибки:")
        for error in error_messages:
            print(error)


if __name__ == "__main__":
    main()