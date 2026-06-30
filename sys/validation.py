import re
import argparse
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple
import time
import wfdb
import numpy as np

class PeakValidator:
    """Валидатор для простого QRS детектора без классификации"""
    
    def __init__(self, annotations_dir: str, peaks_dir: str, results_dir: str):
        self.annotations_dir = Path(annotations_dir)
        self.peaks_dir = Path(peaks_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.tolerance_samples = None
    
    def parse_peaks_file(self, peaks_path: Path) -> Tuple[List[int], float, int]:
        """Парсит файл с обнаруженными пиками (*_peaks.txt)"""
        try:
            with open(peaks_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            total_match = re.search(r'Total Peaks Detected:\s*(\d+)', content)
            total_peaks = int(total_match.group(1)) if total_match else 0
            
            hr_match = re.search(r'Heart Rate:\s*([\d.]+)\s*BPM', content)
            heart_rate = float(hr_match.group(1)) if hr_match else 0.0
            
            peak_line_match = re.search(r'Peak indices \(\d+ peaks\):\s*([\d,\s]+)', content)
            if not peak_line_match:
                return [], heart_rate, total_peaks
            
            peaks_str = peak_line_match.group(1)
            peaks = [int(x.strip()) for x in re.findall(r'\d+', peaks_str)]
            
            return peaks, heart_rate, total_peaks
            
        except Exception as e:
            print(f"Ошибка при парсинге {peaks_path}: {e}")
            return [], 0.0, 0
    
    def get_annotations(self, record_name: str) -> Tuple[np.ndarray, int]:
        """Получает индексы R-пиков из аннотаций"""
        try:
            record_path = self.annotations_dir / record_name
            
            annotation = wfdb.rdann(str(record_path), 'atr')
            peak_indices = np.array(annotation.sample)
            
            record = wfdb.rdrecord(str(record_path))
            fs = record.fs
            
            return peak_indices, fs
            
        except Exception as e:
            print(f"Ошибка при чтении аннотаций для {record_name}: {e}")
            return np.array([]), 0
    
    def evaluate_detection(self, true_peaks: np.ndarray, detected_peaks: List[int], 
                           fs: int) -> Dict[str, float]:
        """Оценка работы QRS детектора"""
        
        tolerance = int(0.150 * fs)
        
        true_set = set(true_peaks)
        detected_set = set(detected_peaks)
        
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
        missed_percent = fn_count / len(true_peaks) * 100 if len(true_peaks) > 0 else 0.0
        
        hr_error = self.calculate_hr_error(true_peaks, detected_peaks, fs)
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'false_positives': fp_count,
            'false_negatives': fn_count,
            'missed_percent': missed_percent,
            'hr_error': hr_error,
            'total_true': len(true_peaks),
            'total_detected': len(detected_peaks)
        }
    
    def calculate_hr_error(self, true_peaks: np.ndarray, detected_peaks: List[int], 
                          fs: int) -> float:
        """СКЗ погрешности ЧСС"""
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
        
        errors = []
        for i in range(min_len):
            errors.append((true_rr[i] - detected_rr[i]) ** 2)
        
        return np.sqrt(np.mean(errors)) if errors else 0.0
    
    def validate_record(self, record_name: str, db_type: str = 'MIT-BIH') -> Dict[str, float]:
        """Выполняет валидацию для одной записи"""
        results = {}
        
        true_peaks, fs = self.get_annotations(record_name)
        if len(true_peaks) == 0:
            return {'error': 'No annotations found'}
        
        peaks_file = self.peaks_dir / f"{record_name}_peaks.txt"
        if not peaks_file.exists():
            return {'error': f'Peaks file not found: {peaks_file}'}
        
        detected_peaks, hr_detected, total_detected = self.parse_peaks_file(peaks_file)
        if not detected_peaks:
            return {'error': 'No peaks detected'}
        
        metrics = self.evaluate_detection(true_peaks, detected_peaks, fs)
        
        results['Чувствительность QRS'] = metrics['sensitivity']
        results['Специфичность QRS'] = metrics['specificity']
        results['Число ложноположительных результатов'] = metrics['false_positives']
        results['СКЗ погрешности ЧСС'] = metrics['hr_error']
        results['Процент пропущенных кардиоциклов при ВЫКЛЮЧЕНИИ'] = metrics['missed_percent']
        
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
        results['Heart_Rate_Detected'] = hr_detected
        results['False_Negatives'] = metrics['false_negatives']
        
        return results
    
    def save_results(self, record_name: str, results: Dict[str, float], 
                     db_name: str, db_type: str = 'MIT-BIH'):
        """Сохраняет результаты в файл"""
        output_file = self.results_dir / f"{record_name}_results.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"{db_name}\n")
            
            if db_type in ['MIT-BIH', 'AHA']:
                param_order = [
                    'Чувствительность QRS',
                    'Специфичность QRS',
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
                    'Процент пропущенных кардиоциклов при ВЫКЛЮЧЕНИИ',
                    'Процент пропущенных N при ВЫКЛЮЧЕНИИ',
                    'Процент пропущенных V при ВЫКЛЮЧЕНИИ',
                    'Процент пропущенных F при ВЫКЛЮЧЕНИИ'
                ]
            else:
                param_order = [
                    'Чувствительность QRS',
                    'Специфичность QRS',
                    'Чувствительность VEB',
                    'Специфичность VEB',
                    'Число ложноположительных результатов',
                    'СКЗ погрешности ЧСС',
                    'Процент пропущенных кардиоциклов при ВЫКЛЮЧЕНИИ',
                    'Процент пропущенных N при ВЫКЛЮЧЕНИИ',
                    'Процент пропущенных V при ВЫКЛЮЧЕНИИ',
                    'Процент пропущенных F при ВЫКЛЮЧЕНИИ'
                ]
            
            for param in param_order:
                if param in results:
                    if param == 'Число ложноположительных результатов':
                        f.write(f"{param}: {int(results[param])}\n")
                    else:
                        f.write(f"{param}: {results[param]:.4f}\n")
            
            f.write(f"\nAdditional Info:\n")
            if 'Total_True_Peaks' in results:
                f.write(f"Total True Peaks: {int(results['Total_True_Peaks'])}\n")
            if 'Total_Detected_Peaks' in results:
                f.write(f"Total Detected Peaks: {int(results['Total_Detected_Peaks'])}\n")
            if 'Heart_Rate_Detected' in results:
                f.write(f"Heart Rate Detected: {results['Heart_Rate_Detected']:.2f} BPM\n")
            if 'False_Negatives' in results:
                f.write(f"False Negatives (Missed): {int(results['False_Negatives'])}\n")
        
        return output_file

def process_record_wrapper(args):
    """Обертка для многопроцессорной обработки"""
    record_name, annotations_dir, peaks_dir, results_dir, db_name, db_type = args
    
    try:
        validator = PeakValidator(annotations_dir, peaks_dir, results_dir)
        results = validator.validate_record(record_name, db_type)
        
        if 'error' in results:
            return (record_name, False, results['error'])
        
        output_file = validator.save_results(record_name, results, db_name, db_type)
        return (record_name, True, str(output_file))
        
    except Exception as e:
        return (record_name, False, str(e))

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
    parser = argparse.ArgumentParser(description='Валидация QRS детектора')
    parser.add_argument('annotations_dir', help='Путь к директории с файлами аннотаций (*.atr)')
    parser.add_argument('peaks_dir', help='Путь к директории с файлами обнаруженных пиков (*_peaks.txt)')
    parser.add_argument('results_dir', help='Путь к директории для сохранения результатов')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Количество параллельных процессов (по умолчанию: число ядер CPU)')
    
    args = parser.parse_args()
    
    annotations_dir = Path(args.annotations_dir)
    peaks_dir = Path(args.peaks_dir)
    results_dir = Path(args.results_dir)
    
    if not annotations_dir.exists() or not annotations_dir.is_dir():
        print(f"Ошибка: Папка с аннотациями {annotations_dir} не найдена")
        return
    
    if not peaks_dir.exists() or not peaks_dir.is_dir():
        print(f"Ошибка: Папка с пиками {peaks_dir} не найдена")
        return
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    db_name = annotations_dir.name
    db_type = detect_database_type(db_name)
    
    atr_files = list(annotations_dir.glob("*.atr"))
    if not atr_files:
        print(f"В папке {annotations_dir} не найдено .atr файлов")
        return
    
    record_names = [f.stem for f in atr_files]
    
    process_args = []
    for record_name in record_names:
        peaks_file = peaks_dir / f"{record_name}_peaks.txt"
        if peaks_file.exists():
            process_args.append((record_name, str(annotations_dir), str(peaks_dir), 
                               str(results_dir), db_name, db_type))
    
    if not process_args:
        print("Не найдено пар (аннотация/пики) для обработки")
        return
    
    total_files = len(process_args)
    cpu_count = multiprocessing.cpu_count()
    max_workers = min(args.workers if args.workers else cpu_count, total_files)
    
    
    start_time = time.time()
    successful = 0
    failed = 0
    completed = 0
    error_messages = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_record_wrapper, args) for args in process_args]
        
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            
            if result[1]:  # успешно
                successful += 1
            else:
                failed += 1
                if result[2]:
                    error_messages.append(f"  - {result[0]}: {result[2]}")
            
            elapsed = time.time() - start_time
            if completed > 0:
                avg_time = elapsed / completed
                remaining = (total_files - completed) * avg_time
                print(f"\rОбработка: {completed}/{total_files} | OK: {successful} | ERR: {failed} | ~{remaining:.0f}с", end="")
    
    print()
    print("-" * 60)
    
    if failed > 0:
        print("\nОшибки:")
        for error in error_messages:
            print(error)

if __name__ == "__main__":
    main()