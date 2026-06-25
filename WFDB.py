import pandas as pd
import numpy as np
from pathlib import Path
import os

def load_reference_annotations(filepath):
    """
    Загрузка референсных аннотаций из файла
    """
    samples = []
    symbols = []
    
    print(f"Загрузка референсных аннотаций из: {filepath}")
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
    
    print(f"  Загружено референсных меток: {len(samples)}")
    return samples, symbols

def load_predictions(filepath):
    """
    Загрузка предсказаний алгоритма из CSV файла
    """
    print(f"Загрузка предсказаний алгоритма из: {filepath}")
    df = pd.read_csv(filepath)
    
    # Проверяем название колонки с пиками
    if 'peak_index' in df.columns:
        samples = df['peak_index'].tolist()
    elif 'peaks' in df.columns:
        samples = df['peaks'].tolist()
    else:
        # Если название неизвестно, берем первую колонку
        samples = df.iloc[:, 0].tolist()
    
    print(f"  Загружено предсказаний алгоритма: {len(samples)}")
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

def validate_files(ref_path, pred_path):
    """
    Основная функция валидации
    """
    # Проверка существования файлов
    if not os.path.exists(ref_path):
        print(f"ОШИБКА: Файл референса не найден: {ref_path}")
        return False
    
    if not os.path.exists(pred_path):
        print(f"ОШИБКА: Файл предсказаний не найден: {pred_path}")
        return False
    
    # Загрузка данных
    ref_samples, ref_symbols = load_reference_annotations(ref_path)
    pred_samples = load_predictions(pred_path)
    
    # Настройка параметров валидации
    FS = 360  # Гц (частота дискретизации для MIT-BIH)
    WINDOW_MS = 150  # миллисекунд
    WINDOW_SAMPLES = int(FS * WINDOW_MS / 1000)
    
    print(f"\nПараметры валидации:")
    print(f"  Частота дискретизации: {FS} Гц")
    print(f"  Окно поиска: {WINDOW_MS} мс = {WINDOW_SAMPLES} сэмплов")
    
    # Запуск валидации
    results = calculate_metrics(ref_samples, pred_samples, WINDOW_SAMPLES)
    
    # Вывод результатов
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ВАЛИДАЦИИ АЛГОРИТМА ДЕТЕКЦИИ R-ЗУБЦОВ")
    print("=" * 60)
    print(f"{'Метрика':<25} {'Значение':<15} {'Процент':<15}")
    print("-" * 60)
    print(f"{'Всего референсных ударов:':<25} {len(ref_samples):<15}")
    print(f"{'Всего обнаружено алгоритмом:':<25} {len(pred_samples):<15}")
    print("-" * 60)
    print(f"{'Верно обнаружено (TP):':<25} {results['tp']:<15} {results['tp']/len(ref_samples)*100:>14.2f}%")
    print(f"{'Ложно-положительные (FP):':<25} {results['fp']:<15} {results['fp']/len(pred_samples)*100:>14.2f}%")
    print(f"{'Ложно-отрицательные (FN):':<25} {results['fn']:<15} {results['fn']/len(ref_samples)*100:>14.2f}%")
    print("-" * 60)
    
    # Вычисляем основные метрики
    precision = results['tp'] / (results['tp'] + results['fp']) if (results['tp'] + results['fp']) > 0 else 0
    recall = results['tp'] / (results['tp'] + results['fn']) if (results['tp'] + results['fn']) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = results['tp'] / len(ref_samples)  # Sensitivity/Recall
    
    print(f"{'Точность (Precision):':<25} {precision:.4f}     {precision*100:>14.2f}%")
    print(f"{'Полнота (Recall/Sensitivity):':<25} {recall:.4f}     {recall*100:>14.2f}%")
    print(f"{'F1-мера:':<25} {f1:.4f}     {f1*100:>14.2f}%")
    print(f"{'Accuracy:':<25} {accuracy:.4f}     {accuracy*100:>14.2f}%")
    print("=" * 60)
    
    # Анализ ошибок по типам ударов
    if results['fn'] > 0:
        print("\n" + "=" * 60)
        print("АНАЛИЗ ПРОПУЩЕННЫХ УДАРОВ (FN) ПО ТИПАМ")
        print("=" * 60)
        
        # Определяем, какие референсы были пропущены
        missed_types = {}
        total_by_type = {}
        
        for i, (sample, symbol) in enumerate(zip(ref_samples, ref_symbols)):
            total_by_type[symbol] = total_by_type.get(symbol, 0) + 1
            if not results['used_ref'][i]:
                missed_types[symbol] = missed_types.get(symbol, 0) + 1
        
        print(f"{'Тип':<10} {'Пропущено':<12} {'Всего':<12} {'Процент':<12}")
        print("-" * 60)
        for symbol in sorted(missed_types.keys()):
            missed = missed_types[symbol]
            total = total_by_type[symbol]
            percentage = (missed / total * 100) if total > 0 else 0
            print(f"{symbol:<10} {missed:<12} {total:<12} {percentage:>11.1f}%")
        
        # Особое внимание на желудочковые экстрасистолы (V)
        if 'V' in missed_types:
            print(f"\nВНИМАНИЕ: Пропущено {missed_types['V']} желудочковых экстрасистол (тип V)")
            print("   Это критично для диагностики аритмий!")
    
    # Анализ ложных срабатываний
    if results['fp'] > 0:
        print("\n" + "=" * 60)
        print("АНАЛИЗ ЛОЖНЫХ СРАБАТЫВАНИЙ (FP)")
        print("=" * 60)
        
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
        
        print(f"Всего ложных срабатываний: {results['fp']}")
        print("\nТоп-10 ложных срабатываний (по расстоянию до ближайшего референса):")
        print(f"{'N':<5} {'Предсказание':<15} {'Ближайший референс':<20} {'Расстояние':<12}")
        print("-" * 60)
        for i, (pred, ref, dist) in enumerate(fp_analysis[:10], 1):
            print(f"{i:<5} {pred:<15} {ref:<20} {dist:<12}")
        
        if results['fp'] > 20:
            print("\nВНИМАНИЕ: Много ложных срабатываний! Возможные причины:")
            print("   - Алгоритм путает T-зубцы с R-зубцами")
            print("   - Шумовые выбросы на ЭКГ")
            print("   - Слишком низкий порог детекции")
    
    # Статистика распределения типов
    print("\n" + "=" * 60)
    print("СТАТИСТИКА РАСПРЕДЕЛЕНИЯ ТИПОВ УДАРОВ")
    print("=" * 60)
    
    type_counts = {}
    for symbol in ref_symbols:
        type_counts[symbol] = type_counts.get(symbol, 0) + 1
    
    print(f"{'Тип':<10} {'Количество':<15} {'Процент':<15}")
    print("-" * 60)
    for symbol in sorted(type_counts.keys()):
        count = type_counts[symbol]
        percentage = (count / len(ref_symbols) * 100)
        print(f"{symbol:<10} {count:<15} {percentage:>14.1f}%")
    
    # Интерпретация результатов
    print("\n" + "=" * 60)
    print("ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ")
    print("=" * 60)
    
    if f1 >= 0.99:
        print("Отлично! Алгоритм показывает выдающиеся результаты (F1 >= 0.99)")
    elif f1 >= 0.95:
        print("Хорошо! Алгоритм показывает высокое качество детекции (F1 >= 0.95)")
    elif f1 >= 0.90:
        print("Удовлетворительно. Алгоритм требует доработки (F1 >= 0.90)")
    else:
        print("Плохо. Алгоритм требует значительной доработки (F1 < 0.90)")
    
    if results['fp'] > results['fn']:
        print("   - Преобладают ложные срабатывания (FP > FN)")
        print("     Рекомендация: повысить порог детекции")
    elif results['fn'] > results['fp']:
        print("   - Преобладают пропуски ударов (FN > FP)")
        print("     Рекомендация: понизить порог детекции")
    else:
        print("   - Баланс между ложными срабатываниями и пропусками")
    
    print("=" * 60)
    print("\nВалидация завершена!")
    return True

def main():
    """
    Главная функция с консольным вводом путей к файлам
    """
    print("=" * 60)
    print("ВАЛИДАЦИЯ АЛГОРИТМА ДЕТЕКЦИИ R-ЗУБЦОВ")
    print("=" * 60)
    
    # Запрос путей к файлам
    print("\nВведите пути к файлам:")
    
    # Путь к файлу референса
    default_ref = "MIH-BIN/100annotations.txt"
    ref_input = input(f"Путь к файлу референса [{default_ref}]: ").strip()
    ref_path = ref_input if ref_input else default_ref
    
    # Путь к файлу предсказаний
    default_pred = "MIH-BIN-P-RESULTS/100-peaks.csv"
    pred_input = input(f"Путь к файлу предсказаний [{default_pred}]: ").strip()
    pred_path = pred_input if pred_input else default_pred
    
    print("\n" + "-" * 60)
    
    # Запуск валидации
    success = validate_files(ref_path, pred_path)
    
    if not success:
        print("\nВалидация не выполнена из-за ошибок.")
        return
    
    # Предложение сохранить результаты
    save = input("\nСохранить результаты в файл? (y/n): ").strip().lower()
    if save == 'y' or save == 'yes':
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"validation_results_{timestamp}.txt"
        
        # Перенаправление вывода в файл (упрощенный вариант)
        import sys
        original_stdout = sys.stdout
        with open(output_file, 'w', encoding='utf-8') as f:
            sys.stdout = f
            # Повторный запуск валидации с выводом в файл
            validate_files(ref_path, pred_path)
        sys.stdout = original_stdout
        
        print(f"Результаты сохранены в файл: {output_file}")

if __name__ == "__main__":
    main()