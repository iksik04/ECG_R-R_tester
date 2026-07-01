import argparse
import multiprocessing
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Set
import wfdb
import numpy as np
from utils import print_progress


class PeakValidator:
    """Валидатор для QRS детектора с использованием аннотаций в формате WFDB"""
    
    # Допустимые символы для QRS-детекции по ГОСТ
    QRS_SYMBOLS = ['N', 'V']  # N - нормальные, V - желудочковые экстрасистолы
    
    # Символы VF-сегментов
    VF_START = '['
    VF_END = ']'
    
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
        По умолчанию оставляет символы 'N' и 'V' (согласно ГОСТ).
        """
        if allowed_symbols is None:
            allowed_symbols = self.QRS_SYMBOLS
        
        if len(peak_indices) != len(symbols):
            return peak_indices
        
        mask = np.isin(symbols, allowed_symbols)
        return peak_indices[mask]
    
    def get_signal_data(self, record_name: str, ann_dir: Path) -> Tuple[Optional[np.ndarray], int, Optional[str]]:
        """
        Получает сигнал и частоту дискретизации из записи WFDB.
        Возвращает: (signal, fs, error_message)
        """
        try:
            record_path = ann_dir / record_name
            record = wfdb.rdrecord(str(record_path))
            signal = record.p_signal[:, 0] if record.p_signal.shape[1] > 0 else None
            return signal, record.fs, None
        except Exception as e:
            return None, 0, f"Ошибка чтения записи {record_name}: {e}"
    
    def get_vf_segments(self, record_name: str) -> List[Tuple[int, int]]:
        """
        Возвращает список сегментов VF (начало, конец) из аннотаций.
        Согласно ГОСТ Р МЭК 60601-2-47, сегменты VF исключаются из поциклового сравнения.
        """
        try:
            annotation = wfdb.rdann(str(self.annotations_dir / record_name), 'atr')
            vf_segments = []
            start = None
            
            for i, sym in enumerate(annotation.symbol):
                if sym == self.VF_START:  # Начало VF
                    start = annotation.sample[i]
                elif sym == self.VF_END and start is not None:  # Конец VF
                    vf_segments.append((start, annotation.sample[i]))
                    start = None
            
            return vf_segments
        except Exception:
            return []
    
    def calculate_tp_fp_fn(self, true_peaks: np.ndarray, detected_peaks: np.ndarray,
                           fs: int, tolerance_ms: float = 150.0) -> Dict[str, any]:
        """
        Рассчитывает TP, FP, FN для обнаружения QRS с использованием толерантности в миллисекундах.
        Согласно ГОСТ Р МЭК 60601-2-47, окно поиска составляет ±150 мс.
        """
        if len(true_peaks) == 0:
            return {
                'tp': 0,
                'fp': len(detected_peaks),
                'fn': 0,
                'matched_true': set(),
                'matched_detected': set()
            }
        
        if len(detected_peaks) == 0:
            return {
                'tp': 0,
                'fp': 0,
                'fn': len(true_peaks),
                'matched_true': set(),
                'matched_detected': set()
            }
        
        tolerance = max(1, int((tolerance_ms / 1000.0) * fs))
        
        tp_count = 0
        matched_true = set()
        matched_detected = set()
        
        # Оптимизированный поиск соответствий
        detected_set = set(detected_peaks)
        
        for true_peak in true_peaks:
            # Ищем ближайший обнаруженный пик в окне толерантности
            min_peak = true_peak - tolerance
            max_peak = true_peak + tolerance
            
            for detected in detected_peaks:
                if min_peak <= detected <= max_peak:
                    if detected not in matched_detected:
                        tp_count += 1
                        matched_true.add(true_peak)
                        matched_detected.add(detected)
                        break
        
        fn_count = len(true_peaks) - len(matched_true)
        fp_count = len(detected_peaks) - len(matched_detected)
        
        return {
            'tp': tp_count,
            'fp': fp_count,
            'fn': fn_count,
            'matched_true': matched_true,
            'matched_detected': matched_detected
        }
    
    def calculate_tn(self, signal: np.ndarray, true_peaks: np.ndarray, detected_peaks: np.ndarray,
                     fs: int, tolerance_ms: float = 150.0, method: str = 'window') -> int:
        """
        Рассчитывает TN (True Negatives) для QRS детекции.
        
        Методы расчета:
        - 'window': разбивает сигнал на окна длиной tolerance_ms (рекомендуемый метод)
        - 'sample': считает каждую выборку как отдельный классификационный случай
        
        По ГОСТ Р МЭК 60601-2-47 рекомендуется оконный метод для клинических систем.
        """
        if len(true_peaks) == 0:
            if method == 'window':
                window_size = max(1, int((tolerance_ms / 1000.0) * fs))
                return len(signal) // window_size
            else:
                return len(signal)
        
        if method == 'window':
            window_size = max(1, int((tolerance_ms / 1000.0) * fs))
            total_windows = len(signal) // window_size
            
            if total_windows == 0:
                return 0
            
            # Определяем окна, содержащие истинные пики
            windows_with_true = set()
            for peak in true_peaks:
                window_idx = min(peak // window_size, total_windows - 1)
                windows_with_true.add(window_idx)
            
            # Определяем окна, содержащие обнаруженные пики
            windows_with_detected = set()
            for peak in detected_peaks:
                window_idx = min(peak // window_size, total_windows - 1)
                windows_with_detected.add(window_idx)
            
            # TN = окна без истинных пиков И без обнаружений
            tn_windows = total_windows - len(windows_with_true | windows_with_detected)
            return tn_windows
            
        elif method == 'sample':
            tolerance = max(1, int((tolerance_ms / 1000.0) * fs))
            
            true_labels = np.zeros(len(signal), dtype=bool)
            pred_labels = np.zeros(len(signal), dtype=bool)
            
            for peak in true_peaks:
                start = max(0, peak - tolerance)
                end = min(len(signal), peak + tolerance + 1)
                true_labels[start:end] = True
            
            for peak in detected_peaks:
                start = max(0, peak - tolerance)
                end = min(len(signal), peak + tolerance + 1)
                pred_labels[start:end] = True
            
            tn_samples = np.sum(~true_labels & ~pred_labels)
            return tn_samples
        
        else:
            raise ValueError(f"Неизвестный метод расчета TN: {method}")
    
    def evaluate_detection_gost(self, record_name: str, true_peaks: np.ndarray, 
                                detected_peaks: np.ndarray, signal: np.ndarray,
                                fs: int, tolerance_ms: float = 150.0) -> Dict[str, float]:
        """
        Оценка работы QRS детектора по ГОСТ Р МЭК 60601-2-47.
        Рассчитывает все необходимые метрики: чувствительность, специфичность,
        FP, FN, TP, TN.
        """
        if len(true_peaks) == 0:
            return {
                'Чувствительность': 0.0,
                'Специфичность': 0.0,
                'Положительное предсказательное значение': 0.0,
                'True Positives (TP)': 0,
                'False Positives (FP)': len(detected_peaks),
                'False Negatives (FN)': 0,
                'True Negatives (TN)': self.calculate_tn(signal, true_peaks, detected_peaks, fs, tolerance_ms),
                'Ложноположительные': len(detected_peaks),
                'Ложноотрицательные': 0,
                'Процент пропущенных': 0.0,
                'Всего истинных пиков': 0,
                'Всего обнаруженных пиков': len(detected_peaks)
            }
        
        if len(detected_peaks) == 0:
            tn = self.calculate_tn(signal, true_peaks, detected_peaks, fs, tolerance_ms)
            return {
                'Чувствительность': 0.0,
                'Специфичность': 1.0 if tn > 0 else 0.0,
                'Положительное предсказательное значение': 0.0,
                'True Positives (TP)': 0,
                'False Positives (FP)': 0,
                'False Negatives (FN)': len(true_peaks),
                'True Negatives (TN)': tn,
                'Ложноположительные': 0,
                'Ложноотрицательные': len(true_peaks),
                'Процент пропущенных': 100.0,
                'Всего истинных пиков': len(true_peaks),
                'Всего обнаруженных пиков': 0
            }
        
        # Рассчитываем TP, FP, FN
        detection_stats = self.calculate_tp_fp_fn(true_peaks, detected_peaks, fs, tolerance_ms)
        tp = detection_stats['tp']
        fp = detection_stats['fp']
        fn = detection_stats['fn']
        
        # Рассчитываем TN
        tn = self.calculate_tn(signal, true_peaks, detected_peaks, fs, tolerance_ms)
        
        # Рассчитываем метрики по ГОСТ
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # +P по ГОСТ
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        missed_percent = fn / len(true_peaks) * 100 if len(true_peaks) > 0 else 0.0
        
        return {
            'Чувствительность': sensitivity,
            'Специфичность': specificity,
            'Положительное предсказательное значение': ppv,
            'True Positives (TP)': tp,
            'False Positives (FP)': fp,
            'False Negatives (FN)': fn,
            'True Negatives (TN)': tn,
            'Ложноположительные': fp,
            'Ложноотрицательные': fn,
            'Процент пропущенных': missed_percent,
            'Всего истинных пиков': len(true_peaks),
            'Всего обнаруженных пиков': len(detected_peaks)
        }
    
    def validate_record(self, record_name: str, 
                       filter_predictions: bool = True,
                       allowed_symbols: List[str] = None,
                       tolerance_ms: float = 150.0,
                       tn_method: str = 'window') -> Dict[str, float]:
        """
        Выполняет валидацию для одной записи по ГОСТ Р МЭК 60601-2-47.
        Сравнивает истинные аннотации с предсказанными.
        
        Включает:
        1. Фильтрацию истинных пиков по символам (только N и V)
        2. Исключение VF-сегментов из поциклового сравнения
        """
        results = {}
        
        # 1. Получаем истинные пики с символами
        true_peaks, true_symbols, error = self.get_peaks_from_annotation(
            record_name, self.annotations_dir, 'atr', return_symbols=True
        )
        if error:
            return {'error': error}
        
        if len(true_peaks) == 0:
            return {'error': 'Истинные пики не найдены'}
        
        # 2. ФИЛЬТРУЕМ ИСТИННЫЕ ПИКИ ПО СИМВОЛАМ (согласно ГОСТ, оставляем только N и V)
        if len(true_peaks) > 0 and len(true_symbols) > 0:
            filter_symbols = allowed_symbols if allowed_symbols else self.QRS_SYMBOLS
            true_peaks = self.filter_peaks_by_symbol(true_peaks, true_symbols, filter_symbols)
        
        if len(true_peaks) == 0:
            return {'error': f'После фильтрации по символам {filter_symbols} не осталось пиков'}
        
        # 3. Получаем сигнал и частоту дискретизации
        signal, fs, signal_error = self.get_signal_data(record_name, self.annotations_dir)
        if signal_error:
            return {'error': signal_error}
        
        if signal is None:
            return {'error': f'Не удалось получить сигнал для {record_name}'}
        
        # 4. Получаем VF-сегменты для исключения из поциклового сравнения
        vf_segments = self.get_vf_segments(record_name)
        
        # 5. ИСКЛЮЧАЕМ VF-СЕГМЕНТЫ ИЗ ИСТИННЫХ ПИКОВ
        if vf_segments and len(true_peaks) > 0:
            mask = np.ones(len(true_peaks), dtype=bool)
            for start, end in vf_segments:
                mask &= ~((true_peaks >= start) & (true_peaks <= end))
            true_peaks = true_peaks[mask]
        
        if len(true_peaks) == 0:
            return {'error': f'После исключения VF-сегментов не осталось пиков'}
        
        # 6. Получаем предсказанные пики с символами
        pred_peaks, pred_symbols, pred_error = self.get_peaks_from_annotation(
            record_name, self.predictions_dir, 'atr', return_symbols=True
        )
        
        if pred_error:
            return {'error': pred_error}
        
        # 7. Применяем фильтрацию к предсказаниям
        if filter_predictions and len(pred_peaks) > 0 and len(pred_symbols) > 0:
            pred_peaks = self.filter_peaks_by_symbol(pred_peaks, pred_symbols, allowed_symbols)
        
        # 8. ИСКЛЮЧАЕМ VF-СЕГМЕНТЫ ИЗ ПРЕДСКАЗАННЫХ ПИКОВ
        if vf_segments and len(pred_peaks) > 0:
            mask = np.ones(len(pred_peaks), dtype=bool)
            for start, end in vf_segments:
                mask &= ~((pred_peaks >= start) & (pred_peaks <= end))
            pred_peaks = pred_peaks[mask]
        
        # 9. Оцениваем качество детекции по ГОСТ
        metrics = self.evaluate_detection_gost(
            record_name, true_peaks, pred_peaks, signal, fs, tolerance_ms
        )
        
        # 10. Формируем результаты
        results['Чувствительность'] = metrics.get('Чувствительность', 0.0)
        results['Специфичность'] = metrics.get('Специфичность', 0.0)
        results['Положительное предсказательное значение'] = metrics.get('Положительное предсказательное значение', 0.0)
        results['True Positives (TP)'] = metrics.get('True Positives (TP)', 0)
        results['False Positives (FP)'] = metrics.get('False Positives (FP)', 0)
        results['False Negatives (FN)'] = metrics.get('False Negatives (FN)', 0)
        results['True Negatives (TN)'] = metrics.get('True Negatives (TN)', 0)
        results['Ложноположительные'] = metrics.get('Ложноположительные', 0)
        results['Ложноотрицательные'] = metrics.get('Ложноотрицательные', 0)
        results['Процент пропущенных'] = metrics.get('Процент пропущенных', 0.0)
        results['Всего истинных пиков'] = metrics.get('Всего истинных пиков', 0)
        results['Всего обнаруженных пиков'] = metrics.get('Всего обнаруженных пиков', 0)
        
        return results
    
    def save_results(self, record_name: str, results: Dict[str, float]) -> Path:
        """Сохраняет результаты в файл согласно ГОСТ Р МЭК 60601-2-47"""
        output_file = self.results_dir / f"{record_name}_results.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Запись: {record_name}\n")
            f.write("=" * 70 + "\n")
            f.write("Результаты валидации по ГОСТ Р МЭК 60601-2-47\n")
            f.write("=" * 70 + "\n\n")
            
            # Основные метрики по ГОСТ
            f.write("ОСНОВНЫЕ МЕТРИКИ (по ГОСТ Р МЭК 60601-2-47):\n")
            f.write("-" * 50 + "\n")
            
            # Чувствительность и специфичность - основные метрики по ГОСТ
            f.write(f"Чувствительность QRS (Se)                     : {results.get('Чувствительность', 0.0):.4f}\n")
            f.write(f"Специфичность QRS (+P)                        : {results.get('Специфичность', 0.0):.4f}\n")
            f.write(f"Положительное предсказательное значение (PPV) : {results.get('Положительное предсказательное значение', 0.0):.4f}\n")
            f.write(f"Процент пропущенных                           : {results.get('Процент пропущенных', 0.0):.2f}%\n")
            
            f.write("\n")
            f.write("МАТРИЦА ОШИБОК:\n")
            f.write("-" * 50 + "\n")
            
            matrix_order = [
                ('True Positives (TP)', '{:d}'),
                ('True Negatives (TN)', '{:d}'),
                ('False Positives (FP)', '{:d}'),
                ('False Negatives (FN)', '{:d}'),
                ('Ложноположительные', '{:d}'),
                ('Ложноотрицательные', '{:d}'),
            ]
            
            for key, fmt in matrix_order:
                if key in results:
                    f.write(f"{key:30}: {fmt.format(int(results[key]))}\n")
            
            f.write("\n")
            f.write("СТАТИСТИКА ПИКОВ:\n")
            f.write("-" * 50 + "\n")
            
            peak_order = [
                ('Всего истинных пиков (после фильтрации)', '{:d}'),
                ('Всего обнаруженных пиков (после фильтрации)', '{:d}'),
            ]
            
            for key, fmt in peak_order:
                if key in results:
                    f.write(f"{key:40}: {fmt.format(int(results[key]))}\n")
            
            f.write("\n")
            f.write("=" * 70 + "\n")
            f.write(f"Метод расчета TN: window (окна по 150 мс)\n")
            f.write(f"Фильтрация пиков: {', '.join(self.QRS_SYMBOLS)} (только N и V)\n")
            f.write(f"VF-сегменты исключены: {'Да' if results.get('Всего истинных пиков', 0) > 0 else 'Нет'}\n")
            f.write("=" * 70 + "\n")
        
        return output_file
    
    def validate_batch(self, record_names: List[str], 
                      filter_predictions: bool = True,
                      allowed_symbols: List[str] = None,
                      tolerance_ms: float = 150.0,
                      tn_method: str = 'window') -> Dict[str, Dict]:
        """
        Выполняет валидацию для списка записей.
        Возвращает словарь с результатами для каждой записи.
        """
        results_dict = {}
        
        for record_name in record_names:
            results = self.validate_record(
                record_name, filter_predictions, allowed_symbols, tolerance_ms, tn_method
            )
            results_dict[record_name] = results
            
            if 'error' not in results:
                self.save_results(record_name, results)
        
        return results_dict
    
    def parse_results_file(self, file_path: Path) -> Dict[str, any]:
        """Парсит файл результатов и возвращает словарь с метриками"""
        results = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Ищем ключевые метрики
                import re
                
                # Чувствительность
                match = re.search(r'Чувствительность QRS.*?:\s*([\d.]+)', content)
                if match:
                    results['Чувствительность'] = float(match.group(1))
                
                # Специфичность
                match = re.search(r'Специфичность QRS.*?:\s*([\d.]+)', content)
                if match:
                    results['Специфичность'] = float(match.group(1))
                
                # PPV
                match = re.search(r'PPV.*?:\s*([\d.]+)', content)
                if match:
                    results['Положительное предсказательное значение'] = float(match.group(1))
                
                # TP, FP, FN, TN - ищем все варианты написания
                match = re.search(r'True Positives \(TP\).*?:\s*(\d+)', content)
                if match:
                    results['True Positives (TP)'] = int(match.group(1))
                
                match = re.search(r'False Positives \(FP\).*?:\s*(\d+)', content)
                if match:
                    results['False Positives (FP)'] = int(match.group(1))
                
                match = re.search(r'False Negatives \(FN\).*?:\s*(\d+)', content)
                if match:
                    results['False Negatives (FN)'] = int(match.group(1))
                
                match = re.search(r'True Negatives \(TN\).*?:\s*(\d+)', content)
                if match:
                    results['True Negatives (TN)'] = int(match.group(1))
                
                # Ищем "Всего истинных пиков" и "Всего обнаруженных пиков"
                # Возможные варианты ключей:
                # - "Всего истинных пиков (после фильтрации)"
                # - "Всего истинных пиков"
                match = re.search(r'Всего истинных пиков[^:]*:\s*(\d+)', content)
                if match:
                    results['Всего истинных пиков'] = int(match.group(1))
                else:
                    # Если не нашли, вычисляем из TP и FN
                    tp = results.get('True Positives (TP)', 0)
                    fn = results.get('False Negatives (FN)', 0)
                    results['Всего истинных пиков'] = tp + fn
                
                match = re.search(r'Всего обнаруженных пиков[^:]*:\s*(\d+)', content)
                if match:
                    results['Всего обнаруженных пиков'] = int(match.group(1))
                else:
                    # Если не нашли, вычисляем из TP и FP
                    tp = results.get('True Positives (TP)', 0)
                    fp = results.get('False Positives (FP)', 0)
                    results['Всего обнаруженных пиков'] = tp + fp
                
        except Exception as e:
            print(f"Ошибка парсинга {file_path}: {e}")
        
        return results
    
    def generate_summary_report(self, record_names: List[str]) -> Path:
        """
         Генерирует сводный отчет по всем записям.
        """
        summary_file = self.results_dir / "summary_report.txt"
        
        # Собираем статистику по всем записям
        total_metrics = {
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'tn': 0,
            'true_peaks': 0,
            'detected_peaks': 0,
            'sensitivities': [],
            'specificities': [],
            'ppvs': []
        }
        
        successful_records = []
        failed_records = []
        
        for record_name in record_names:
            result_file = self.results_dir / f"{record_name}_results.txt"
            if result_file.exists():
                results = self.parse_results_file(result_file)
                if results:
                    successful_records.append(record_name)
                    
                    # Получаем значения с проверкой на существование
                    tp = results.get('True Positives (TP)', 0)
                    fp = results.get('False Positives (FP)', 0)
                    fn = results.get('False Negatives (FN)', 0)
                    tn = results.get('True Negatives (TN)', 0)
                    
                    total_metrics['tp'] += tp
                    total_metrics['fp'] += fp
                    total_metrics['fn'] += fn
                    total_metrics['tn'] += tn
                    
                    # Если в results нет этих полей, вычисляем их
                    true_peaks = results.get('Всего истинных пиков', tp + fn)
                    detected_peaks = results.get('Всего обнаруженных пиков', tp + fp)
                    
                    total_metrics['true_peaks'] += true_peaks
                    total_metrics['detected_peaks'] += detected_peaks
                    
                    total_metrics['sensitivities'].append(results.get('Чувствительность', 0.0))
                    total_metrics['specificities'].append(results.get('Специфичность', 0.0))
                    total_metrics['ppvs'].append(results.get('Положительное предсказательное значение', 0.0))
                else:
                    failed_records.append(record_name)
            else:
                failed_records.append(record_name)
        
        # Рассчитываем общие метрики
        tp = total_metrics['tp']
        fp = total_metrics['fp']
        fn = total_metrics['fn']
        tn = total_metrics['tn']
        
        overall_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        overall_specificity = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        overall_ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Средние значения по записям
        avg_sensitivity = np.mean(total_metrics['sensitivities']) if total_metrics['sensitivities'] else 0.0
        avg_specificity = np.mean(total_metrics['specificities']) if total_metrics['specificities'] else 0.0
        avg_ppv = np.mean(total_metrics['ppvs']) if total_metrics['ppvs'] else 0.0
        
        # Сохраняем сводный отчет
        with open(summary_file, 'w', encoding='utf-8') as f:
            
            f.write(f"Всего записей: {len(record_names)}\n")
            f.write(f"Успешно обработано: {len(successful_records)}\n")
            f.write(f"С ошибками: {len(failed_records)}\n\n")
            
            if failed_records:
                f.write("ЗАПИСИ С ОШИБКАМИ:\n")
                f.write("-" * 50 + "\n")
                for rec in failed_records:
                    f.write(f"  - {rec}\n")
                f.write("\n")
            
            f.write("ОБЩИЕ МЕТРИКИ (по всем записям):\n")
            f.write("-" * 60 + "\n")
            f.write(f"Общая чувствительность QRS (Se)        : {overall_sensitivity:.4f}\n")
            f.write(f"Общая специфичность QRS (+P)           : {overall_specificity:.4f}\n")
            f.write(f"Общее PPV                              : {overall_ppv:.4f}\n\n")
            
            f.write("СРЕДНИЕ МЕТРИКИ (по записям):\n")
            f.write("-" * 60 + "\n")
            f.write(f"Средняя чувствительность               : {avg_sensitivity:.4f}\n")
            f.write(f"Средняя специфичность                  : {avg_specificity:.4f}\n")
            f.write(f"Среднее PPV                            : {avg_ppv:.4f}\n\n")
            
            f.write("ОБЩАЯ МАТРИЦА ОШИБОК:\n")
            f.write("-" * 60 + "\n")
            f.write(f"True Positives (TP): {tp}\n")
            f.write(f"True Negatives (TN): {tn}\n")
            f.write(f"False Positives (FP): {fp}\n")
            f.write(f"False Negatives (FN): {fn}\n\n")
            
            f.write("СТАТИСТИКА ПИКОВ:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Всего истинных пиков: {total_metrics['true_peaks']}\n")
            f.write(f"Всего обнаруженных пиков: {total_metrics['detected_peaks']}\n")
            f.write(f"Разница: {total_metrics['detected_peaks'] - total_metrics['true_peaks']}\n\n")
            
            f.write("ПОЗАПИСНАЯ СТАТИСТИКА:\n")
            f.write("-" * 60 + "\n")
            
            for record_name in successful_records:
                result_file = self.results_dir / f"{record_name}_results.txt"
                results = self.parse_results_file(result_file)
                if results:
                    f.write(f"\n{record_name}:\n")
                    f.write(f"  Чувствительность: {results.get('Чувствительность', 0.0):.4f}\n")
                    f.write(f"  Специфичность:    {results.get('Специфичность', 0.0):.4f}\n")
                    f.write(f"  PPV:              {results.get('Положительное предсказательное значение', 0.0):.4f}\n")
                    f.write(f"  TP: {results.get('True Positives (TP)', 0):d}, "
                           f"FP: {results.get('False Positives (FP)', 0):d}, "
                           f"FN: {results.get('False Negatives (FN)', 0):d}, "
                           f"TN: {results.get('True Negatives (TN)', 0):d}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("ПРИМЕЧАНИЯ:\n")
            f.write("-" * 60 + "\n")
            f.write("1. Фильтрация пиков: только символы N и V (согласно ГОСТ)\n")
            f.write("2. VF-сегменты исключены из поциклового сравнения\n")
            f.write("3. Метод расчета TN: оконный (окна по 150 мс)\n")
            f.write("=" * 80 + "\n")
        
        return summary_file


def process_record_wrapper(args: Tuple) -> Tuple[str, bool, str]:
    """Обертка для многопроцессорной обработки"""
    record_name, annotations_dir, predictions_dir, results_dir, filter_pred, tolerance_ms, symbols, tn_method = args
    
    try:
        validator = PeakValidator(annotations_dir, predictions_dir, results_dir)
        results = validator.validate_record(record_name, filter_pred, symbols, tolerance_ms, tn_method)
        
        if 'error' in results:
            return (record_name, False, f"Ошибка валидации: {results['error']}")
        
        output_file = validator.save_results(record_name, results)
        return (record_name, True, str(output_file))
        
    except Exception as e:
        return (record_name, False, f"Исключение: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description='Валидация QRS детектора по ГОСТ Р МЭК 60601-2-47'
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
                        help='Толерантность в миллисекундах (по умолчанию: 150 мс, согласно ГОСТ)')
    parser.add_argument('--symbols', type=str, default='N,V',
                        help='Символы для фильтрации через запятую (по умолчанию: N,V)')
    parser.add_argument('--tn-method', type=str, default='window',
                        choices=['window', 'sample'],
                        help='Метод расчета TN: window (окна) или sample (поэлементно)')
    parser.add_argument('--summary', action='store_true',
                        help='Сгенерировать сводный отчет после обработки всех записей')
    
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
                allowed_symbols,
                args.tn_method
            ))
        else:
            print(f"Предупреждение: Нет предсказаний для {record_name}")
    
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
    
    elapsed_time = time.time() - start_time
    
    # Вывод ошибок если они были
    if failed > 0:
        print("\nОШИБКИ:")
        for error in error_messages:
            print(error)
    
    # Генерация сводного отчета
    if args.summary and successful > 0:
        validator = PeakValidator(str(annotations_dir), str(predictions_dir), str(results_dir))
        summary_file = validator.generate_summary_report(record_names)


if __name__ == "__main__":
    main()