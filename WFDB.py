from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import multiprocessing
import re

def load_reference_annotations(filepath):
    """
    Загрузка референсных аннотаций из файла
    """
    samples = []
    symbols = []
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            
            # Пропускаем заголовок
            try:
                sample = int(parts[1])  # Колонка Num (индекс сэмпла)
            except ValueError:
                continue
            
            symbol = parts[2]  # Колонка Type (N, A, V, + и т.д.)
            
            # Игнорируем служебные символы (начало сегмента)
            if symbol == '+':
                continue
            
            samples.append(sample)
            symbols.append(symbol)
    
    return samples, symbols

def load_predictions(filepath):
    """
    Загрузка предсказаний алгоритма из файла с пиками
    Поддерживает формат с заголовком и секцией "Peak indices"
    """
    samples = []
    
    with open(filepath, 'r') as f:
        content = f.read()
        
        # Ищем секцию с пиками
        # Формат: "Peak indices (X peaks):" или просто список чисел
        match = re.search(r'Peak indices\s*\([^)]+\):\s*([\d,\s]+)', content)
        
        if match:
            # Извлекаем числа из найденной секции
            numbers_str = match.group(1)
            # Разбиваем по запятым и пробелам, фильтруем пустые строки
            for num_str in re.findall(r'\d+', numbers_str):
                try:
                    samples.append(int(num_str))
                except ValueError:
                    continue
        else:
            # Если секция не найдена, пробуем прочитать все числа из файла
            # Пропускаем строки, начинающиеся с неподходящих слов
            for line in content.split('\n'):
                line = line.strip()
                if not line or line.startswith('Results for:') or line.startswith('===') or line.startswith('Total') or line.startswith('Heart'):
                    continue
                # Ищем числа в строке
                for num_str in re.findall(r'\d+', line):
                    try:
                        samples.append(int(num_str))
                    except ValueError:
                        continue
    
    return samples

def calculate_tp(ref, pred, window):
    """
    Расчет True Positives (TP)
    """
    ref = sorted(ref)
    pred = sorted(pred)
    used = [False] * len(ref)
    tp = 0
    tp_samples = []
    
    for p in pred:
        min_dist = float('inf')
        best_idx = -1
        
        for i, r in enumerate(ref):
            if used[i]:
                continue
            dist = abs(p - r)
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        
        if best_idx != -1 and min_dist <= window:
            tp += 1
            used[best_idx] = True
            tp_samples.append((p, ref[best_idx], min_dist))
    
    return tp, used, tp_samples

def calculate_fp(pred, ref, used, window):
    """
    Расчет False Positives (FP)
    """
    fp = 0
    fp_samples = []
    
    for p in pred:
        min_dist = float('inf')
        best_idx = -1
        
        for i, r in enumerate(ref):
            dist = abs(p - r)
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        
        # Если не использован или расстояние больше окна - это FP
        if best_idx == -1 or min_dist > window:
            fp += 1
            fp_samples.append(p)
        else:
            # Проверяем, использован ли этот референс
            if not used[best_idx]:
                fp += 1
                fp_samples.append(p)
    
    return fp, fp_samples

def calculate_fn(ref, used):
    """
    Расчет False Negatives (FN)
    """
    fn = 0
    fn_samples = []
    
    for i, u in enumerate(used):
        if not u:
            fn += 1
            fn_samples.append(ref[i])
    
    return fn, fn_samples

def calculate_precision(tp, fp):
    """Расчет точности (Precision)"""
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def calculate_recall(tp, fn):
    """Расчет полноты (Recall/Sensitivity)"""
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def calculate_f1(precision, recall):
    """Расчет F1-меры"""
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

def calculate_metrics(ref, pred, window):
    """
    Расчет всех метрик для детекции R-зубцов
    """
    ref = sorted(ref)
    pred = sorted(pred)
    
    # Расчет TP
    tp, used, tp_samples = calculate_tp(ref, pred, window)
    
    # Расчет FP
    fp, fp_samples = calculate_fp(pred, ref, used, window)
    
    # Расчет FN
    fn, fn_samples = calculate_fn(ref, used)
    
    # Расчет производных метрик
    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    f1 = calculate_f1(precision, recall)
    accuracy = recall  # Accuracy = Sensitivity = Recall
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp_samples': tp_samples,
        'fp_samples': fp_samples,
        'fn_samples': fn_samples,
        'used_ref': used,
        'total_ref': len(ref),
        'total_pred': len(pred)
    }

def generate_report(ref_path, pred_path, results):
    """
    Генерация отчета по результатам валидации
    """
    # Загружаем символы для анализа по типам
    ref_samples, ref_symbols = load_reference_annotations(ref_path)
    
    lines = []
    lines.append("=" * 60)
    lines.append("РЕЗУЛЬТАТЫ ВАЛИДАЦИИ АЛГОРИТМА ДЕТЕКЦИИ R-ЗУБЦОВ")
    lines.append("=" * 60)
    lines.append(f"{'Метрика':<25} {'Значение':<15} {'Процент':<15}")
    lines.append("-" * 60)
    lines.append(f"{'Всего референсных ударов:':<25} {results['total_ref']:<15}")
    lines.append(f"{'Всего обнаружено алгоритмом:':<25} {results['total_pred']:<15}")
    lines.append("-" * 60)
    lines.append(f"{'Верно обнаружено (TP):':<25} {results['tp']:<15} {results['tp']/results['total_ref']*100:>14.2f}%")
    lines.append(f"{'Ложно-положительные (FP):':<25} {results['fp']:<15} {results['fp']/results['total_pred']*100 if results['total_pred'] > 0 else 0:>14.2f}%")
    lines.append(f"{'Ложно-отрицательные (FN):':<25} {results['fn']:<15} {results['fn']/results['total_ref']*100:>14.2f}%")
    lines.append("-" * 60)
    lines.append(f"{'Точность (Precision):':<25} {results['precision']:.4f}     {results['precision']*100:>14.2f}%")
    lines.append(f"{'Полнота (Recall/Sensitivity):':<25} {results['recall']:.4f}     {results['recall']*100:>14.2f}%")
    lines.append(f"{'F1-мера:':<25} {results['f1']:.4f}     {results['f1']*100:>14.2f}%")
    lines.append(f"{'Accuracy:':<25} {results['accuracy']:.4f}     {results['accuracy']*100:>14.2f}%")
    lines.append("=" * 60)
    
    # Анализ пропущенных ударов по типам
    if results['fn'] > 0:
        lines.append("\n" + "=" * 60)
        lines.append("АНАЛИЗ ПРОПУЩЕННЫХ УДАРОВ (FN) ПО ТИПАМ")
        lines.append("=" * 60)
        
        missed_types = {}
        total_by_type = {}
        
        for i, (sample, symbol) in enumerate(zip(ref_samples, ref_symbols)):
            total_by_type[symbol] = total_by_type.get(symbol, 0) + 1
            if not results['used_ref'][i]:
                missed_types[symbol] = missed_types.get(symbol, 0) + 1
        
        lines.append(f"{'Тип':<10} {'Пропущено':<12} {'Всего':<12} {'Процент':<12}")
        lines.append("-" * 60)
        for symbol in sorted(missed_types.keys()):
            missed = missed_types[symbol]
            total = total_by_type[symbol]
            percentage = (missed / total * 100) if total > 0 else 0
            lines.append(f"{symbol:<10} {missed:<12} {total:<12} {percentage:>11.1f}%")
        
        if 'V' in missed_types:
            lines.append(f"\nВНИМАНИЕ: Пропущено {missed_types['V']} желудочковых экстрасистол (тип V)")
            lines.append("   Это критично для диагностики аритмий!")
    
    # Анализ ложных срабатываний
    if results['fp'] > 0:
        lines.append("\n" + "=" * 60)
        lines.append("АНАЛИЗ ЛОЖНЫХ СРАБАТЫВАНИЙ (FP)")
        lines.append("=" * 60)
        
        fp_analysis = []
        for fp_sample in results['fp_samples']:
            min_dist = float('inf')
            nearest_ref = -1
            for ref_sample in ref_samples:
                dist = abs(fp_sample - ref_sample)
                if dist < min_dist:
                    min_dist = dist
                    nearest_ref = ref_sample
            fp_analysis.append((fp_sample, nearest_ref, min_dist))
        
        fp_analysis.sort(key=lambda x: x[2])
        
        lines.append(f"Всего ложных срабатываний: {results['fp']}")
        lines.append("\nТоп-10 ложных срабатываний (по расстоянию до ближайшего референса):")
        lines.append(f"{'N':<5} {'Предсказание':<15} {'Ближайший референс':<20} {'Расстояние':<12}")
        lines.append("-" * 60)
        for i, (pred, ref, dist) in enumerate(fp_analysis[:10], 1):
            lines.append(f"{i:<5} {pred:<15} {ref:<20} {dist:<12}")
        
        if results['fp'] > 20:
            lines.append("\nВНИМАНИЕ: Много ложных срабатываний! Возможные причины:")
            lines.append("   - Алгоритм путает T-зубцы с R-зубцами")
            lines.append("   - Шумовые выбросы на ЭКГ")
            lines.append("   - Слишком низкий порог детекции")
    
    # Статистика распределения типов
    lines.append("\n" + "=" * 60)
    lines.append("СТАТИСТИКА РАСПРЕДЕЛЕНИЯ ТИПОВ УДАРОВ")
    lines.append("=" * 60)
    
    type_counts = {}
    for symbol in ref_symbols:
        type_counts[symbol] = type_counts.get(symbol, 0) + 1
    
    lines.append(f"{'Тип':<10} {'Количество':<15} {'Процент':<15}")
    lines.append("-" * 60)
    for symbol in sorted(type_counts.keys()):
        count = type_counts[symbol]
        percentage = (count / len(ref_symbols) * 100)
        lines.append(f"{symbol:<10} {count:<15} {percentage:>14.1f}%")
    
    # Интерпретация результатов
    lines.append("\n" + "=" * 60)
    lines.append("ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ")
    lines.append("=" * 60)
    
    f1 = results['f1']
    if f1 >= 0.99:
        lines.append("Отлично! Алгоритм показывает выдающиеся результаты (F1 >= 0.99)")
    elif f1 >= 0.95:
        lines.append("Хорошо! Алгоритм показывает высокое качество детекции (F1 >= 0.95)")
    elif f1 >= 0.90:
        lines.append("Удовлетворительно. Алгоритм требует доработки (F1 >= 0.90)")
    else:
        lines.append("Плохо. Алгоритм требует значительной доработки (F1 < 0.90)")
    
    if results['fp'] > results['fn']:
        lines.append("   - Преобладают ложные срабатывания (FP > FN)")
        lines.append("     Рекомендация: повысить порог детекции")
    elif results['fn'] > results['fp']:
        lines.append("   - Преобладают пропуски ударов (FN > FP)")
        lines.append("     Рекомендация: понизить порог детекции")
    else:
        lines.append("   - Баланс между ложными срабатываниями и пропусками")
    
    lines.append("=" * 60)
    lines.append("\nВалидация завершена!")
    
    return lines

def process_file_wrapper(args):
    """
    Обёртка для использования с ProcessPoolExecutor
    """
    ref_file, pred_file, output_file = args
    
    try:
        # Параметры валидации
        FS = 360  # Гц (частота дискретизации для MIT-BIH)
        WINDOW_MS = 150  # миллисекунд
        WINDOW_SAMPLES = int(FS * WINDOW_MS / 1000)
        
        # Загрузка данных
        ref_samples, ref_symbols = load_reference_annotations(ref_file)
        pred_samples = load_predictions(pred_file)
        
        # Расчет метрик
        results = calculate_metrics(ref_samples, pred_samples, WINDOW_SAMPLES)
        
        # Генерация отчета
        report_lines = generate_report(ref_file, pred_file, results)
        
        # Сохранение отчета
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in report_lines:
                f.write(line + '\n')
        
        return (Path(ref_file).name, True, results['f1'], None)
    
    except Exception as e:
        return (Path(ref_file).name, False, 0, str(e))

def main():
    """
    Главная функция с парсингом аргументов командной строки
    """
    parser = argparse.ArgumentParser(
        description='Валидация алгоритма детекции R-зубцов на ЭКГ (многопоточная версия)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Примеры использования:
  py WFDB.py DATA/ANNOTATIONS DATA/DETECTED-PEAKS DATA/RESULTS
  py WFDB.py -a DATA/ANNOTATIONS -p DATA/DETECTED-PEAKS -o DATA/RESULTS
  py WFDB.py --annotations DATA/ANNOTATIONS --peaks DATA/DETECTED-PEAKS --output DATA/RESULTS
        '''
    )
    
    # Позиционные аргументы
    parser.add_argument('annotations_dir', 
                        help='Путь к директории с файлами референсных аннотаций')
    parser.add_argument('peaks_dir', 
                        help='Путь к директории с файлами предсказаний алгоритма')
    parser.add_argument('output_dir', 
                        help='Путь к директории для сохранения результатов')
    
    # Альтернативные опции
    parser.add_argument('-a', '--annotations', dest='annotations_alt',
                        help='Альтернативный путь к директории с аннотациями')
    parser.add_argument('-p', '--peaks', dest='peaks_alt',
                        help='Альтернативный путь к директории с пиками')
    parser.add_argument('-o', '--output', dest='output_alt',
                        help='Альтернативный путь к директории результатов')
    
    # Опция для количества процессов
    parser.add_argument('--workers', '-w',
                        type=int,
                        default=None,
                        help='Количество параллельных процессов (по умолчанию: количество ядер CPU)')
    
    args = parser.parse_args()
    
    # Определяем пути (приоритет у альтернативных опций)
    annotations_dir = Path(args.annotations_alt if args.annotations_alt else args.annotations_dir)
    peaks_dir = Path(args.peaks_alt if args.peaks_alt else args.peaks_dir)
    output_dir = Path(args.output_alt if args.output_alt else args.output_dir)
    
    # Проверяем существование директорий
    if not annotations_dir.exists():
        print(f"Ошибка: Папка с аннотациями {annotations_dir} не найдена")
        return
    
    if not annotations_dir.is_dir():
        print(f"Ошибка: {annotations_dir} не является директорией")
        return
    
    if not peaks_dir.exists():
        print(f"Ошибка: Папка с пиками {peaks_dir} не найдена")
        return
    
    if not peaks_dir.is_dir():
        print(f"Ошибка: {peaks_dir} не является директорией")
        return
    
    # Создаем выходную папку
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Находим все файлы аннотаций
    annotation_files = list(annotations_dir.glob("*annotations.txt"))
    
    if not annotation_files:
        print(f"В папке {annotations_dir} не найдено файлов аннотаций (*annotations.txt)")
        return
    
    # Определяем количество процессов
    cpu_count = multiprocessing.cpu_count()
    if args.workers:
        max_workers = min(args.workers, len(annotation_files))
    else:
        max_workers = min(cpu_count * 2, len(annotation_files))
    
    # Подготавливаем аргументы для обработки
    process_args = []
    for ann_file in annotation_files:
        # Получаем базовое имя файла (убираем 'annotations' и расширение)
        stem = ann_file.stem  # 100annotations
        if stem.endswith('annotations'):
            base_name = stem[:-11]  # убираем 'annotations' (11 символов)
        else:
            base_name = stem
        
        # Ищем соответствующий файл с пиками
        possible_peak_files = [
            peaks_dir / f"{base_name}_peaks.txt",
            peaks_dir / f"{base_name}_peaks.csv",
            peaks_dir / f"{base_name}.txt",
            peaks_dir / f"{base_name}.csv",
        ]
        
        peak_file = None
        for p in possible_peak_files:
            if p.exists():
                peak_file = p
                break
        
        if peak_file is None:
            # Пропускаем, если нет соответствующего файла с пиками
            continue
        
        output_file = output_dir / f"{base_name}_results.txt"
        process_args.append((str(ann_file), str(peak_file), str(output_file)))
    
    if not process_args:
        print("Не найдено соответствующих пар файлов (аннотация + пики)")
        return
    
    total_pairs = len(process_args)
    start_time = time.time()
    
    results = []
    completed = 0
    successful = 0
    failed = 0
    error_messages = []
    
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
                if result[3]:
                    error_messages.append(f"  - {result[0]}: {result[3]}")
            
            elapsed = time.time() - start_time
            if completed > 0:
                avg_time = elapsed / completed
                remaining = (total_pairs - completed) * avg_time
                print(f"\rОбработка: {completed}/{total_pairs} | OK: {successful} | ERR: {failed} | ~{remaining:.0f}с", end="")
    
    print()
    
    # Вывод ошибок если они были
    if failed > 0:
        print("\nОшибки:")
        for error in error_messages:
            print(error)

if __name__ == "__main__":
    main()