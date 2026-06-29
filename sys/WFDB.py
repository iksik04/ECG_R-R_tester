import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
import argparse
from datetime import datetime

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
    Загрузка предсказаний алгоритма из CSV или TXT файла
    """
    # Определяем формат файла по расширению
    file_ext = Path(filepath).suffix.lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(filepath)
        # Проверяем название колонки с пиками
        if 'peak_index' in df.columns:
            samples = df['peak_index'].tolist()
        elif 'peaks' in df.columns:
            samples = df['peaks'].tolist()
        else:
            # Если название неизвестно, берем первую колонку
            samples = df.iloc[:, 0].tolist()
    else:
        # Пробуем загрузить как текстовый файл с числами
        samples = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        # Может быть одно число или несколько через разделители
                        parts = line.replace(',', ' ').replace(';', ' ').split()
                        for part in parts:
                            samples.append(int(float(part)))
                    except ValueError:
                        continue
    
    return samples

def calculate_metrics(ref, pred, window):
    """
    Расчет метрик для детекции R-зубцов
    
    Parameters:
    ref: список референсных сэмплов
    pred: список предсказанных сэмплов
    window: допустимое отклонение в сэмплах
    
    Returns:
    dict: TP, FP, FN, а также списки ошибок
    """
    ref = sorted(ref)
    pred = sorted(pred)
    
    tp = 0
    fp = 0
    fn = 0
    
    # Для отслеживания использованных референсов
    used = [False] * len(ref)
    fp_samples = []
    fn_samples = []
    tp_samples = []
    
    # Для каждого предсказания ищем ближайший референс
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
        
        # Если нашли в пределах окна - это TP
        if best_idx != -1 and min_dist <= window:
            tp += 1
            used[best_idx] = True
            tp_samples.append((p, ref[best_idx], min_dist))
        else:
            fp += 1
            fp_samples.append(p)
    
    # Неиспользованные референсы - это FN
    for i, u in enumerate(used):
        if not u:
            fn += 1
            fn_samples.append(ref[i])
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tp_samples': tp_samples,
        'fp_samples': fp_samples,
        'fn_samples': fn_samples,
        'used_ref': used
    }

def validate_files(ref_path, pred_path, output_file=None):
    """
    Основная функция валидации
    """
    # Проверка существования файлов
    if not os.path.exists(ref_path):
        return False
    
    if not os.path.exists(pred_path):
        return False
    
    # Загрузка данных
    ref_samples, ref_symbols = load_reference_annotations(ref_path)
    pred_samples = load_predictions(pred_path)
    
    # Настройка параметров валидации
    FS = 360  # Гц (частота дискретизации для MIT-BIH)
    WINDOW_MS = 150  # миллисекунд
    WINDOW_SAMPLES = int(FS * WINDOW_MS / 1000)
    
    # Запуск валидации
    results = calculate_metrics(ref_samples, pred_samples, WINDOW_SAMPLES)
    
    # Формируем строки с результатами
    output_lines = []
    output_lines.append("=" * 60)
    output_lines.append("РЕЗУЛЬТАТЫ ВАЛИДАЦИИ АЛГОРИТМА ДЕТЕКЦИИ R-ЗУБЦОВ")
    output_lines.append("=" * 60)
    output_lines.append(f"{'Метрика':<25} {'Значение':<15} {'Процент':<15}")
    output_lines.append("-" * 60)
    output_lines.append(f"{'Всего референсных ударов:':<25} {len(ref_samples):<15}")
    output_lines.append(f"{'Всего обнаружено алгоритмом:':<25} {len(pred_samples):<15}")
    output_lines.append("-" * 60)
    output_lines.append(f"{'Верно обнаружено (TP):':<25} {results['tp']:<15} {results['tp']/len(ref_samples)*100:>14.2f}%")
    output_lines.append(f"{'Ложно-положительные (FP):':<25} {results['fp']:<15} {results['fp']/len(pred_samples)*100:>14.2f}%")
    output_lines.append(f"{'Ложно-отрицательные (FN):':<25} {results['fn']:<15} {results['fn']/len(ref_samples)*100:>14.2f}%")
    output_lines.append("-" * 60)
    
    # Вычисляем основные метрики
    precision = results['tp'] / (results['tp'] + results['fp']) if (results['tp'] + results['fp']) > 0 else 0
    recall = results['tp'] / (results['tp'] + results['fn']) if (results['tp'] + results['fn']) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = results['tp'] / len(ref_samples)  # Sensitivity/Recall
    
    output_lines.append(f"{'Точность (Precision):':<25} {precision:.4f}     {precision*100:>14.2f}%")
    output_lines.append(f"{'Полнота (Recall/Sensitivity):':<25} {recall:.4f}     {recall*100:>14.2f}%")
    output_lines.append(f"{'F1-мера:':<25} {f1:.4f}     {f1*100:>14.2f}%")
    output_lines.append(f"{'Accuracy:':<25} {accuracy:.4f}     {accuracy*100:>14.2f}%")
    output_lines.append("=" * 60)
    
    # Анализ ошибок по типам ударов
    if results['fn'] > 0:
        output_lines.append("\n" + "=" * 60)
        output_lines.append("АНАЛИЗ ПРОПУЩЕННЫХ УДАРОВ (FN) ПО ТИПАМ")
        output_lines.append("=" * 60)
        
        # Определяем, какие референсы были пропущены
        missed_types = {}
        total_by_type = {}
        
        for i, (sample, symbol) in enumerate(zip(ref_samples, ref_symbols)):
            total_by_type[symbol] = total_by_type.get(symbol, 0) + 1
            if not results['used_ref'][i]:
                missed_types[symbol] = missed_types.get(symbol, 0) + 1
        
        output_lines.append(f"{'Тип':<10} {'Пропущено':<12} {'Всего':<12} {'Процент':<12}")
        output_lines.append("-" * 60)
        for symbol in sorted(missed_types.keys()):
            missed = missed_types[symbol]
            total = total_by_type[symbol]
            percentage = (missed / total * 100) if total > 0 else 0
            output_lines.append(f"{symbol:<10} {missed:<12} {total:<12} {percentage:>11.1f}%")
        
        # Особое внимание на желудочковые экстрасистолы (V)
        if 'V' in missed_types:
            output_lines.append(f"\nВНИМАНИЕ: Пропущено {missed_types['V']} желудочковых экстрасистол (тип V)")
            output_lines.append("   Это критично для диагностики аритмий!")
    
    # Анализ ложных срабатываний
    if results['fp'] > 0:
        output_lines.append("\n" + "=" * 60)
        output_lines.append("АНАЛИЗ ЛОЖНЫХ СРАБАТЫВАНИЙ (FP)")
        output_lines.append("=" * 60)
        
        # Находим ближайшие референсы к ложным срабатываниям
        fp_analysis = []
        for fp_sample in results['fp_samples']:
            # Ищем ближайший референс (даже если он уже использован)
            min_dist = float('inf')
            nearest_ref = -1
            for ref_sample in ref_samples:
                dist = abs(fp_sample - ref_sample)
                if dist < min_dist:
                    min_dist = dist
                    nearest_ref = ref_sample
            fp_analysis.append((fp_sample, nearest_ref, min_dist))
        
        # Сортируем по расстоянию до ближайшего референса
        fp_analysis.sort(key=lambda x: x[2])
        
        output_lines.append(f"Всего ложных срабатываний: {results['fp']}")
        output_lines.append("\nТоп-10 ложных срабатываний (по расстоянию до ближайшего референса):")
        output_lines.append(f"{'N':<5} {'Предсказание':<15} {'Ближайший референс':<20} {'Расстояние':<12}")
        output_lines.append("-" * 60)
        for i, (pred, ref, dist) in enumerate(fp_analysis[:10], 1):
            output_lines.append(f"{i:<5} {pred:<15} {ref:<20} {dist:<12}")
        
        if results['fp'] > 20:
            output_lines.append("\nВНИМАНИЕ: Много ложных срабатываний! Возможные причины:")
            output_lines.append("   - Алгоритм путает T-зубцы с R-зубцами")
            output_lines.append("   - Шумовые выбросы на ЭКГ")
            output_lines.append("   - Слишком низкий порог детекции")
    
    # Статистика распределения типов
    output_lines.append("\n" + "=" * 60)
    output_lines.append("СТАТИСТИКА РАСПРЕДЕЛЕНИЯ ТИПОВ УДАРОВ")
    output_lines.append("=" * 60)
    
    type_counts = {}
    for symbol in ref_symbols:
        type_counts[symbol] = type_counts.get(symbol, 0) + 1
    
    output_lines.append(f"{'Тип':<10} {'Количество':<15} {'Процент':<15}")
    output_lines.append("-" * 60)
    for symbol in sorted(type_counts.keys()):
        count = type_counts[symbol]
        percentage = (count / len(ref_symbols) * 100)
        output_lines.append(f"{symbol:<10} {count:<15} {percentage:>14.1f}%")
    
    # Интерпретация результатов
    output_lines.append("\n" + "=" * 60)
    output_lines.append("ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ")
    output_lines.append("=" * 60)
    
    if f1 >= 0.99:
        output_lines.append("Отлично! Алгоритм показывает выдающиеся результаты (F1 >= 0.99)")
    elif f1 >= 0.95:
        output_lines.append("Хорошо! Алгоритм показывает высокое качество детекции (F1 >= 0.95)")
    elif f1 >= 0.90:
        output_lines.append("Удовлетворительно. Алгоритм требует доработки (F1 >= 0.90)")
    else:
        output_lines.append("Плохо. Алгоритм требует значительной доработки (F1 < 0.90)")
    
    if results['fp'] > results['fn']:
        output_lines.append("   - Преобладают ложные срабатывания (FP > FN)")
        output_lines.append("     Рекомендация: повысить порог детекции")
    elif results['fn'] > results['fp']:
        output_lines.append("   - Преобладают пропуски ударов (FN > FP)")
        output_lines.append("     Рекомендация: понизить порог детекции")
    else:
        output_lines.append("   - Баланс между ложными срабатываниями и пропусками")
    
    output_lines.append("=" * 60)
    output_lines.append("\nВалидация завершена!")
    
    # Сохраняем в файл, если указан
    if output_file:
        try:
            # Создаем директорию для выходного файла, если её нет
            output_dir = Path(output_file).parent
            if output_dir and not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for line in output_lines:
                    f.write(line + '\n')
        except Exception as e:
            return False
    
    return True

def main():
    """
    Главная функция с парсингом аргументов командной строки
    """
    parser = argparse.ArgumentParser(
        description='Валидация алгоритма детекции R-зубцов на ЭКГ',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Примеры использования:
  py WFDB.py DATA/DATABASES/MIH-BIN/100annotations.txt DATA/DETECTED-PEAKS/MIH-BIN-PEAKS/100_peaks.txt DATA/RESULTS/100_results.txt
  py WFDB.py -r ref.txt -p pred.txt -o results.txt
  py WFDB.py ref.txt pred.txt  # без сохранения в файл
        '''
    )
    
    # Позиционные аргументы
    parser.add_argument('ref_file', 
                        help='Путь к файлу референсных аннотаций')
    parser.add_argument('pred_file', 
                        help='Путь к файлу с предсказаниями алгоритма')
    parser.add_argument('output_file', nargs='?', default=None,
                        help='Путь к файлу для сохранения результатов (опционально)')
    
    # Дополнительные опции
    parser.add_argument('-r', '--ref', dest='ref_alt',
                        help='Альтернативный путь к файлу референса')
    parser.add_argument('-p', '--pred', dest='pred_alt',
                        help='Альтернативный путь к файлу предсказаний')
    parser.add_argument('-o', '--output', dest='output_alt',
                        help='Альтернативный путь к файлу результатов')
    
    args = parser.parse_args()
    
    # Определяем пути к файлам (приоритет у альтернативных опций)
    ref_path = args.ref_alt if args.ref_alt else args.ref_file
    pred_path = args.pred_alt if args.pred_alt else args.pred_file
    output_path = args.output_alt if args.output_alt else args.output_file
    
    # Запуск валидации
    success = validate_files(ref_path, pred_path, output_path)
    
    if not success:
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main()