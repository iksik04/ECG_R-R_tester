import argparse
import multiprocessing
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import wfdb
import numpy as np
from utils import print_progress


class PeakValidator:
    """Валидатор для QRS детектора с использованием аннотаций в формате WFDB"""
    
    def __init__(self, annotations_dir: str, predictions_dir: str, results_dir: str):
        self.annotations_dir = Path(annotations_dir)
        self.predictions_dir = Path(predictions_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def get_peaks_from_annotation(self, record_name: str, ann_dir: Path, 
                                  ann_ext: str = 'atr', 
                                  return_symbols: bool = False) -> Tuple[np.ndarray, List[str], Optional[str]]:
        """
        Получает индексы пиков и символы из аннотаций WFDB.
        Возвращает: (peak_indices, symbols, error_message)
        """
        try:
            ann_file = ann_dir / f"{record_name}.{ann_ext}"
            if not ann_file.exists():
                return np.array([]), [], f"Файл аннотаций не найден: {ann_file}"
            
            # Читаем аннотацию
            annotation = wfdb.rdann(str(ann_dir / record_name), ann_ext)
            peak_indices = np.array(annotation.sample)
            symbols = list(annotation.symbol) if return_symbols else []
            
            return peak_indices, symbols, None
            
        except Exception as e:
            return np.array([]), [], f"Ошибка чтения аннотации {record_name}: {e}"
    
    def filter_peaks_by_symbol(self, peak_indices: np.ndarray, 
                               symbols: List[str],
                               allowed_symbols: List[str] = None) -> np.ndarray:
        """
        Фильтрует пики по символам аннотаций.
        По умолчанию оставляет символы 'N' и 'V'.
        """
        if allowed_symbols is None:
            allowed_symbols = ['N', 'V']
        
        if len(peak_indices) != len(symbols):
            return peak_indices
        
        mask = np.isin(symbols, allowed_symbols)
        return peak_indices[mask]
    
    def evaluate_detection(self, true_peaks: np.ndarray, detected_peaks: np.ndarray,
                           fs: int, tolerance_ms: float = 150.0) -> Dict[str, float]:
        """
        Оценка работы QRS детектора с использованием толерантности в миллисекундах
        """
        if len(true_peaks) == 0:
            return {
                'Чувствительность': 0.0,
                'Положительное предсказательное значение': 0.0,
                'Ложноположительные': 0,
                'Ложноотрицательные': 0,
                'Процент пропущенных': 100.0,
                'Всего истинных пиков': 0,
                'Всего обнаруженных пиков': len(detected_peaks)
            }
        
        if len(detected_peaks) == 0:
            return {
                'Чувствительность': 0.0,
                'Положительное предсказательное значение': 0.0,
                'Ложноположительные': 0,
                'Ложноотрицательные': len(true_peaks),
                'Процент пропущенных': 100.0,
                'Всего истинных пиков': len(true_peaks),
                'Всего обнаруженных пиков': 0
            }
        
        tolerance = int((tolerance_ms / 1000.0) * fs)
        
        tp_count = 0
        matched_true = set()
        matched_detected = set()
        
        for detected in detected_peaks:
            for true in true_peaks:
                if abs(detected - true) <= tolerance:
                    tp_count += 1
                    matched_true.add(true)
                    matched_detected.add(detected)
                    break
        
        fn_count = len(true_peaks) - len(matched_true)
        fp_count = len(detected_peaks) - len(matched_detected)
        
        sensitivity = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        positive_predictive_value = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        missed_percent = fn_count / len(true_peaks) * 100 if len(true_peaks) > 0 else 0.0
        
        return {
            'Чувствительность': sensitivity,
            'Положительное предсказательное значение': positive_predictive_value,
            'Ложноположительные': fp_count,
            'Ложноотрицательные': fn_count,
            'Процент пропущенных': missed_percent,
            'Всего истинных пиков': len(true_peaks),
            'Всего обнаруженных пиков': len(detected_peaks)
        }
    
    def validate_record(self, record_name: str, 
                       filter_predictions: bool = True,
                       allowed_symbols: List[str] = None,
                       tolerance_ms: float = 150.0) -> Dict[str, float]:
        """
        Выполняет валидацию для одной записи.
        Сравнивает истинные аннотации с предсказанными.
        """
        results = {}
        
        # Получаем истинные пики с символами
        true_peaks, true_symbols, error = self.get_peaks_from_annotation(
            record_name, self.annotations_dir, 'atr', return_symbols=True
        )
        if error:
            return {'error': error}
        
        if len(true_peaks) == 0:
            return {'error': 'Истинные пики не найдены'}
        
        # Получаем частоту дискретизации из записи
        try:
            record_path = self.annotations_dir / record_name
            record = wfdb.rdrecord(str(record_path))
            fs = record.fs
        except Exception as e:
            return {'error': f"Ошибка чтения записи {record_name}: {e}"}
        
        # Получаем предсказанные пики с символами
        pred_peaks, pred_symbols, pred_error = self.get_peaks_from_annotation(
            record_name, self.predictions_dir, 'atr', return_symbols=True
        )
        
        if pred_error:
            return {'error': pred_error}
        
        # Применяем фильтрацию к предсказаниям
        if filter_predictions and len(pred_peaks) > 0 and len(pred_symbols) > 0:
            pred_peaks = self.filter_peaks_by_symbol(pred_peaks, pred_symbols, allowed_symbols)
        
        # Оцениваем качество детекции
        metrics = self.evaluate_detection(true_peaks, pred_peaks, fs, tolerance_ms)
        
        # Формируем результаты
        results['Чувствительность'] = metrics['Чувствительность']
        results['Положительное предсказательное значение'] = metrics['Положительное предсказательное значение']
        results['Ложноположительные'] = metrics['Ложноположительные']
        results['Ложноотрицательные'] = metrics['Ложноотрицательные']
        results['Процент пропущенных'] = metrics['Процент пропущенных']
        results['Всего истинных пиков'] = metrics['Всего истинных пиков']
        results['Всего обнаруженных пиков'] = metrics['Всего обнаруженных пиков']
        
        return results
    
    def save_results(self, record_name: str, results: Dict[str, float]) -> Path:
        """Сохраняет результаты в файл"""
        output_file = self.results_dir / f"{record_name}_results.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Запись: {record_name}\n")
            f.write("-" * 50 + "\n")
            
            # Основные метрики
            param_order = [
                ('Чувствительность', '{:.4f}'),
                ('Положительное предсказательное значение', '{:.4f}'),
                ('Ложноположительные', '{:d}'),
                ('Ложноотрицательные', '{:d}'),
                ('Процент пропущенных', '{:.2f}'),
                ('Всего истинных пиков', '{:d}'),
                ('Всего обнаруженных пиков', '{:d}')
            ]
            
            for key, fmt in param_order:
                if key in results:
                    if 'd' in fmt:  # целые числа
                        f.write(f"{key}: {fmt.format(int(results[key]))}\n")
                    else:
                        f.write(f"{key}: {fmt.format(results[key])}\n")
            
            f.write("-" * 50 + "\n")
        
        return output_file


def process_record_wrapper(args: Tuple) -> Tuple[str, bool, str]:
    """Обертка для многопроцессорной обработки"""
    record_name, annotations_dir, predictions_dir, results_dir, filter_pred, tolerance_ms, symbols = args
    
    try:
        validator = PeakValidator(annotations_dir, predictions_dir, results_dir)
        results = validator.validate_record(record_name, filter_pred, symbols, tolerance_ms)
        
        if 'error' in results:
            return (record_name, False, f"Ошибка валидации: {results['error']}")
        
        output_file = validator.save_results(record_name, results)
        return (record_name, True, str(output_file))
        
    except Exception as e:
        return (record_name, False, f"Исключение: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description='Валидация QRS детектора по аннотациям в формате WFDB (*.atr)'
    )
    parser.add_argument('annotations_dir',
                        help='Путь к директории с истинными аннотациями (*.atr)')
    parser.add_argument('predictions_dir',
                        help='Путь к директории с предсказанными аннотациями (*.atr)')
    parser.add_argument('results_dir',
                        help='Путь к директории для сохранения результатов')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Количество параллельных процессов (по умолчанию: количество ядер CPU)')
    parser.add_argument('--no-filter', action='store_true',
                        help='Не фильтровать предсказания по символам N/V')
    parser.add_argument('--tolerance', type=float, default=150.0,
                        help='Толерантность в миллисекундах (по умолчанию: 150 мс)')
    parser.add_argument('--symbols', type=str, default='N,V',
                        help='Символы для фильтрации через запятую (по умолчанию: N,V)')
    
    args = parser.parse_args()
    
    annotations_dir = Path(args.annotations_dir)
    predictions_dir = Path(args.predictions_dir)
    results_dir = Path(args.results_dir)
    
    if not annotations_dir.exists() or not annotations_dir.is_dir():
        print(f"Ошибка: Папка с аннотациями {annotations_dir} не найдена")
        return
    
    if not predictions_dir.exists() or not predictions_dir.is_dir():
        print(f"Ошибка: Папка с предсказаниями {predictions_dir} не найдена")
        return
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Парсим символы для фильтрации
    allowed_symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
    
    # Находим все файлы аннотаций в папке с истинными данными
    atr_files = list(annotations_dir.glob("*.atr"))
    if not atr_files:
        print(f"В папке {annotations_dir} не найдено .atr файлов")
        return
    
    record_names = [f.stem for f in atr_files]
    
    # Собираем аргументы для обработки
    process_args = []
    for record_name in record_names:
        # Проверяем наличие предсказаний
        pred_file = predictions_dir / f"{record_name}.atr"
        if pred_file.exists():
            process_args.append((
                record_name,
                str(annotations_dir),
                str(predictions_dir),
                str(results_dir),
                not args.no_filter,
                args.tolerance,
                allowed_symbols
            ))
        else:
            print(f"Предупреждение: Нет предсказаний для {record_name} (файл {pred_file} не найден)")
    
    if not process_args:
        print("Не найдено пар (истинные аннотации/предсказания) для обработки")
        return
    
    total_files = len(process_args)
    max_workers = min(args.workers if args.workers else multiprocessing.cpu_count(), total_files)
    
    start_time = time.time()
    completed = 0
    successful = 0
    failed = 0
    error_messages = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_record_wrapper, arg) for arg in process_args]
        
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            
            if result[1]:  # успешно
                successful += 1
            else:
                failed += 1
                if result[2]:
                    error_messages.append(f"  - {result[0]}: {result[2]}")
            
            print_progress(completed, total_files, successful, failed, start_time)
    
    print()  # Переход на новую строку после завершения
    
    # Вывод ошибок если они были
    if failed > 0:
        print("\nОшибки:")
        for error in error_messages:
            print(error)


if __name__ == "__main__":
    main()