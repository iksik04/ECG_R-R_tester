import argparse
import multiprocessing
import time
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import wfdb
import numpy as np


class PeakValidator:
    """Валидатор для QRS детектора с использованием аннотаций в формате WFDB"""
    
    def __init__(self, annotations_dir: str, predictions_dir: str, results_dir: str):
        self.annotations_dir = Path(annotations_dir)
        self.predictions_dir = Path(predictions_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._fs_cache = {}  # Кэш для частоты дискретизации
    
    def get_record_info(self, record_name: str, ann_dir: Path) -> Tuple[int, Optional[str]]:
        """Получает частоту дискретизации из заголовка записи"""
        try:
            record_path = ann_dir / record_name
            record = wfdb.rdrecord(str(record_path))
            return record.fs, None
        except Exception as e:
            return 0, f"Error reading record {record_name}: {e}"
    
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
                return np.array([]), [], f"Annotation file not found: {ann_file}"
            
            # Читаем аннотацию
            annotation = wfdb.rdann(str(ann_dir / record_name), ann_ext)
            peak_indices = np.array(annotation.sample)
            symbols = list(annotation.symbol) if return_symbols else []
            
            return peak_indices, symbols, None
            
        except Exception as e:
            return np.array([]), [], f"Error reading annotation {record_name}: {e}"
    
    def filter_peaks_by_symbol(self, peak_indices: np.ndarray, 
                               symbols: List[str],
                               allowed_symbols: List[str] = None) -> np.ndarray:
        """
        Фильтрует пики по символам аннотаций.
        """
        if allowed_symbols is None:
            allowed_symbols = ['N', 'V']
        
        if len(peak_indices) != len(symbols):
            return peak_indices
        
        mask = np.isin(symbols, allowed_symbols)
        return peak_indices[mask]
    
    def evaluate_detection(self, true_peaks: np.ndarray, detected_peaks: np.ndarray,
                           fs: int) -> Dict[str, float]:
        """
        Оценка работы QRS детектора с использованием толерантности 150 мс
        """
        if len(true_peaks) == 0:
            return {
                'sensitivity': 0.0,
                'specificity': 0.0,
                'positive_predictive_value': 0.0,
                'false_positives': 0,
                'false_negatives': 0,
                'missed_percent': 100.0,
                'hr_error': 0.0,
                'total_true': 0,
                'total_detected': len(detected_peaks)
            }
        
        if len(detected_peaks) == 0:
            return {
                'sensitivity': 0.0,
                'specificity': 0.0,
                'positive_predictive_value': 0.0,
                'false_positives': 0,
                'false_negatives': len(true_peaks),
                'missed_percent': 100.0,
                'hr_error': 0.0,
                'total_true': len(true_peaks),
                'total_detected': 0
            }
        
        tolerance = int(0.150 * fs)
        
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
        specificity = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        positive_predictive_value = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        missed_percent = fn_count / len(true_peaks) * 100 if len(true_peaks) > 0 else 0.0
        
        hr_error = self.calculate_hr_error(true_peaks, detected_peaks, fs)
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'positive_predictive_value': positive_predictive_value,
            'false_positives': fp_count,
            'false_negatives': fn_count,
            'missed_percent': missed_percent,
            'hr_error': hr_error,
            'total_true': len(true_peaks),
            'total_detected': len(detected_peaks)
        }
    
    def calculate_hr_error(self, true_peaks: np.ndarray, detected_peaks: np.ndarray,
                          fs: int) -> float:
        """Вычисляет СКЗ погрешности ЧСС"""
        if len(true_peaks) < 2 or len(detected_peaks) < 2:
            return 0.0
        
        true_rr = []
        sorted_true = sorted(true_peaks)
        for i in range(1, len(sorted_true)):
            true_rr.append((sorted_true[i] - sorted_true[i-1]) / fs)
        
        detected_rr = []
        sorted_detected = sorted(detected_peaks)
        for i in range(1, len(sorted_detected)):
            detected_rr.append((sorted_detected[i] - sorted_detected[i-1]) / fs)
        
        min_len = min(len(true_rr), len(detected_rr))
        if min_len == 0:
            return 0.0
        
        errors = [(true_rr[i] - detected_rr[i]) ** 2 for i in range(min_len)]
        
        return np.sqrt(np.mean(errors)) if errors else 0.0
    
    def calculate_heart_rate(self, peaks: np.ndarray, fs: int) -> float:
        """Вычисляет ЧСС по пикам"""
        if len(peaks) < 2:
            return 0.0
        
        sorted_peaks = sorted(peaks)
        rr_intervals = []
        for i in range(1, len(sorted_peaks)):
            rr_intervals.append((sorted_peaks[i] - sorted_peaks[i-1]) / fs)
        
        if not rr_intervals:
            return 0.0
        
        mean_rr = np.mean(rr_intervals)
        return 60.0 / mean_rr if mean_rr > 0 else 0.0
    
    def validate_record(self, record_name: str, db_type: str = 'MIT-BIH',
                       filter_predictions: bool = True) -> Dict[str, float]:
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
            return {'error': 'No ground truth peaks found'}
        
        # Получаем частоту дискретизации из записи
        fs, fs_error = self.get_record_info(record_name, self.annotations_dir)
        if fs_error:
            return {'error': fs_error}
        
        if fs == 0:
            return {'error': 'Invalid sampling rate'}
        
        # Получаем предсказанные пики с символами
        pred_peaks, pred_symbols, pred_error = self.get_peaks_from_annotation(
            record_name, self.predictions_dir, 'atr', return_symbols=True
        )
        
        if pred_error:
            return {'error': pred_error}
        
        # Применяем фильтрацию к предсказаниям (оставляем только N и V)
        if filter_predictions and len(pred_peaks) > 0 and len(pred_symbols) > 0:
            pred_peaks = self.filter_peaks_by_symbol(
                pred_peaks, pred_symbols, allowed_symbols=['N', 'V']
            )
        
        # Оцениваем качество детекции
        metrics = self.evaluate_detection(true_peaks, pred_peaks, fs)
        
        # Вычисляем ЧСС по предсказаниям
        hr_detected = self.calculate_heart_rate(pred_peaks, fs)
        hr_true = self.calculate_heart_rate(true_peaks, fs)
        
        # Формируем результаты
        results['Чувствительность QRS'] = metrics['sensitivity']
        results['Специфичность QRS'] = metrics['specificity']
        results['Положительное предсказательное значение'] = metrics['positive_predictive_value']
        results['Число ложноположительных результатов'] = metrics['false_positives']
        results['СКЗ погрешности ЧСС'] = metrics['hr_error']
        results['Процент пропущенных кардиоциклов'] = metrics['missed_percent']
        
        # Заглушки для VEB и других параметров
        results['Чувствительность VEB'] = 0.0
        results['Специфичность VEB'] = 0.0
        
        if db_type in ['MIT-BIH', 'AHA']:
            results['Чувствительность к желудочковому куплету'] = 0.0
            results['Специфичность желудочкового куплета'] = 0.0
            results['Чувствительность к желудочковой короткой серии'] = 0.0
            results['Специфичность желудочковой короткой серии'] = 0.0
            results['Чувствительность к желудочковой длинной серии'] = 0.0
            results['Специфичность желудочковой длинной серии'] = 0.0
        
        results['Total_True_Peaks'] = metrics['total_true']
        results['Total_Detected_Peaks'] = metrics['total_detected']
        results['Heart_Rate_True'] = hr_true
        results['Heart_Rate_Detected'] = hr_detected
        results['False_Negatives'] = metrics['false_negatives']
        
        return results
    
    def save_results(self, record_name: str, results: Dict[str, float],
                     db_name: str, db_type: str = 'MIT-BIH') -> Path:
        """Сохраняет результаты в файл"""
        output_file = self.results_dir / f"{record_name}_results.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Database: {db_name}\n")
            f.write(f"Record: {record_name}\n")
            f.write(f"Type: {db_type}\n")
            f.write("-" * 60 + "\n")
            
            if db_type in ['MIT-BIH', 'AHA']:
                param_order = [
                    'Чувствительность QRS',
                    'Специфичность QRS',
                    'Положительное предсказательное значение',
                    'Чувствительность VEB',
                    'Специфичность VEB',
                    'Число ложноположительных результатов',
                    'СКЗ погрешности ЧСС',
                    'Чувствительность к желудочковому куплету',
                    'Специфичность желудочкового куплета',
                    'Чувствительность к желудочковой короткой серии',
                    'Специфичность желудочковой короткой серии',
                    'Чувствительность к желудочковой длинной серии',
                    'Специфичность желудочковой длинной серии',
                    'Процент пропущенных кардиоциклов',
                    'Процент пропущенных N при ВЫКЛЮЧЕНИИ',
                    'Процент пропущенных V при ВЫКЛЮЧЕНИИ',
                    'Процент пропущенных F при ВЫКЛЮЧЕНИИ'
                ]
            else:
                param_order = [
                    'Чувствительность QRS',
                    'Специфичность QRS',
                    'Положительное предсказательное значение',
                    'Чувствительность VEB',
                    'Специфичность VEB',
                    'Число ложноположительных результатов',
                    'СКЗ погрешности ЧСС',
                    'Процент пропущенных кардиоциклов',
                    'Процент пропущенных N при ВЫКЛЮЧЕНИИ',
                    'Процент пропущенных V при ВЫКЛЮЧЕНИИ',
                    'Процент пропущенных F при ВЫКЛЮЧЕНИИ'
                ]
            
            for param in param_order:
                if param in results:
                    if param in ['Число ложноположительных результатов', 'False_Negatives']:
                        f.write(f"{param}: {int(results[param])}\n")
                    else:
                        f.write(f"{param}: {results[param]:.4f}\n")
            
            f.write("-" * 60 + "\n")
            f.write("Additional Info:\n")
            if 'Total_True_Peaks' in results:
                f.write(f"  Total True Peaks: {int(results['Total_True_Peaks'])}\n")
            if 'Total_Detected_Peaks' in results:
                f.write(f"  Total Detected Peaks: {int(results['Total_Detected_Peaks'])}\n")
            if 'Heart_Rate_True' in results and results['Heart_Rate_True'] > 0:
                f.write(f"  Heart Rate (True): {results['Heart_Rate_True']:.2f} BPM\n")
            if 'Heart_Rate_Detected' in results and results['Heart_Rate_Detected'] > 0:
                f.write(f"  Heart Rate (Detected): {results['Heart_Rate_Detected']:.2f} BPM\n")
            if 'False_Negatives' in results:
                f.write(f"  False Negatives (Missed): {int(results['False_Negatives'])}\n")
        
        return output_file


def print_progress(completed, total, successful, failed, start_time):
    """Печатает прогресс-бар на одной строке."""
    elapsed = time.time() - start_time
    avg_time = elapsed / completed if completed > 0 else 0
    remaining = (total - completed) * avg_time if avg_time > 0 else 0
    
    # Ширина прогресс-бара
    bar_width = 40
    filled = int(bar_width * completed / total)
    bar = '█' * filled + '░' * (bar_width - filled)
    
    # Форматируем время
    elapsed_str = f"{elapsed:.0f}s"
    remaining_str = f"{remaining:.0f}s" if remaining > 0 else "0s"
    
    # Строка прогресса
    progress_str = (f"\r[{bar}] {completed}/{total} "
                    f"| OK {successful} ERR {failed} "
                    f"| ~{remaining_str} сек")
    
    sys.stdout.write(progress_str)
    sys.stdout.flush()


def process_record_wrapper(args: Tuple) -> Tuple[str, bool, str]:
    """Обертка для многопроцессорной обработки"""
    record_name, annotations_dir, predictions_dir, results_dir, db_name, db_type, filter_pred = args
    
    try:
        validator = PeakValidator(annotations_dir, predictions_dir, results_dir)
        results = validator.validate_record(record_name, db_type, filter_pred)
        
        if 'error' in results:
            return (record_name, False, f"Validation error: {results['error']}")
        
        output_file = validator.save_results(record_name, results, db_name, db_type)
        return (record_name, True, str(output_file))
        
    except Exception as e:
        return (record_name, False, f"Exception: {str(e)}")


def detect_database_type(db_name: str) -> str:
    """Определяет тип БД по имени"""
    db_name_upper = db_name.upper()
    if 'MIT' in db_name_upper or 'BIH' in db_name_upper:
        return 'MIT-BIH'
    elif 'AHA' in db_name_upper:
        return 'AHA'
    elif 'NST' in db_name_upper:
        return 'NST'
    else:
        return 'MIT-BIH'


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
    
    db_name = annotations_dir.name
    db_type = detect_database_type(db_name)
    
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
                db_name,
                db_type,
                not args.no_filter
            ))
        else:
            print(f"Предупреждение: Нет предсказаний для {record_name} (файл {pred_file} не найден)")
    
    if not process_args:
        print("Не найдено пар (истинные аннотации/предсказания) для обработки")
        return
    
    total_files = len(process_args)
    cpu_count = multiprocessing.cpu_count()
    
    # Определяем количество процессов
    if args.workers:
        max_workers = min(args.workers, total_files)
    else:
        max_workers = min(cpu_count * 2, total_files)
    
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
            
            # Используем обновленный прогресс-бар
            print_progress(completed, total_files, successful, failed, start_time)
    
    print()  # Переход на новую строку после завершения
    
    # Вывод ошибок если они были
    if failed > 0:
        print("\n Ошибки:")
        for error in error_messages:
            print(error)


if __name__ == "__main__":
    main()