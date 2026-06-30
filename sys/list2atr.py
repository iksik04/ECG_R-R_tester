import sys
import os
import wfdb
import numpy as np
from datetime import datetime

def parse_peaks_string(peaks_str):
    """
    Парсит строку с пиками, разделёнными пробелами
    Возвращает список целых чисел
    """
    if not peaks_str or peaks_str.strip() == '':
        return []
    
    try:
        peaks = [int(x) for x in peaks_str.strip().split() if x]
        return peaks
    except ValueError as e:
        print(f"Ошибка парсинга пиков: {e}", file=sys.stderr)
        return []

def save_atr_file(peaks, output_path, sampling_freq=250, description=""):
    """
    Сохраняет пики в формате .atr с использованием библиотеки wfdb
    """
    output_dir = os.path.dirname(output_path) or '.'
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    
    # Создаём директорию, если она не существует
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Сортируем пики
    peaks_sorted = sorted(peaks)
    
    # Создаём массив numpy для пиков
    peak_array = np.array(peaks_sorted, dtype=np.int32)
    
    # Создаём символы аннотаций (по умолчанию 'N' - нормальный удар)
    symbols = np.array(['N'] * len(peak_array), dtype='U1')
    
    # Сохраняем аннотации в формате WFDB
    try:
        wfdb.wrann(
            record_name=base_name,
            extension='atr',
            sample=peak_array,
            symbol=symbols,
            write_dir=output_dir
        )
        
        # Проверяем, что файл создан
        expected_atr = os.path.join(output_dir, f"{base_name}.atr")
        
        # Если путь отличается от ожидаемого, переименовываем
        if expected_atr != output_path:
            if os.path.exists(output_path):
                os.remove(output_path)
            
            if os.path.exists(expected_atr):
                os.rename(expected_atr, output_path)
                
                expected_ari = os.path.join(output_dir, f"{base_name}.ari")
                if os.path.exists(expected_ari):
                    ari_path = output_path.replace('.atr', '.ari')
                    if os.path.exists(ari_path):
                        os.remove(ari_path)
                    os.rename(expected_ari, ari_path)
        
        return len(peaks_sorted)
        
    except FileNotFoundError as e:
        raise Exception(f"Ошибка: директория '{output_dir}' не найдена или недоступна. {e}")
    except PermissionError as e:
        raise Exception(f"Ошибка доступа: нет прав на запись в '{output_dir}'. {e}")
    except Exception as e:
        raise Exception(f"Ошибка при записи аннотаций с помощью wfdb: {e}")

def main():
    # Проверяем аргументы командной строки
    if len(sys.argv) < 2:
        print("Ошибка: необходимо указать путь к выходному .atr файлу", file=sys.stderr)
        print("Пример: python list2atr.py output.atr", file=sys.stderr)
        print("Строка с пиками передаётся через stdin или как аргумент", file=sys.stderr)
        sys.exit(1)
    
    output_file = sys.argv[1]
    
    # Проверяем расширение файла
    if not output_file.lower().endswith('.atr'):
        output_file += '.atr'
    
    # Читаем строку с пиками
    peaks_str = ""
    
    if len(sys.argv) > 2:
        peaks_str = " ".join(sys.argv[2:])
    else:
        peaks_str = sys.stdin.read().strip()
    
    # Парсим пики
    peaks = parse_peaks_string(peaks_str)
    
    if not peaks:
        print("Предупреждение: список пиков пуст. Создаётся файл без пиков.", file=sys.stderr)
    
    # === ДОБАВЛЕННАЯ ЧАСТЬ: СОХРАНЯЕМ ПИКИ ===
    try:
        count = save_atr_file(peaks, output_file)
    except Exception as e:
        print(f"Ошибка при сохранении: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()