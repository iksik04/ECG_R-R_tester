import matplotlib.pyplot as plt
import numpy as np
import re

# Чтение файла с пиками (100_peaks.txt)
def read_peaks(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    # Извлекаем все числа из списка пиков
    numbers = re.findall(r'\b\d+\b', content)
    # Первые несколько чисел - это метаданные (Total Peaks: 2268, Heart Rate: 75.51)
    # Пропускаем их и берем только индексы пиков
    peaks = [int(num) for num in numbers[2:]]  # пропускаем 2268 и 75.51
    return peaks

# Чтение файла с данными ЭКГ (100.dart)
def read_ecg_data(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    # Ищем список данных в формате List<double> data = [...]
    match = re.search(r'List<double> data = \[(.*?)\];', content, re.DOTALL)
    if not match:
        raise ValueError("Не найден список данных в файле")
    
    data_str = match.group(1)
    # Извлекаем все числа с плавающей точкой
    numbers = re.findall(r'-?\d+\.?\d*', data_str)
    data = [float(num) for num in numbers]
    return data

# Чтение файлов
try:
    peaks = read_peaks('DATA/DETECTED-PEAKS/MIT-BIH-PEAKS/100_peaks.txt')
    ecg_data = read_ecg_data('DATA/DART-DATA/MIT-BIH-DART/100.dart')
    
    print(f"Найдено пиков: {len(peaks)}")
    print(f"Длина ЭКГ-сигнала: {len(ecg_data)} точек")
    
    # Создаем график
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Основной график - весь сигнал
    time = np.arange(len(ecg_data)) / 360.0  # перевод в секунды (частота 360 Гц)
    ax1.plot(time, ecg_data, 'b-', linewidth=0.8, label='ECG сигнал')
    
    # Отмечаем пики на графике
    for peak in peaks:
        if peak < len(ecg_data):
            ax1.axvline(x=peak/360.0, color='r', alpha=0.3, linewidth=0.5)
    
    ax1.set_xlabel('Время (сек)')
    ax1.set_ylabel('Амплитуда (мВ)')
    ax1.set_title('Полный ЭКГ-сигнал с отмеченными пиками')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Увеличенный фрагмент для лучшей визуализации
    # Показываем первые 10 секунд
    end_time = min(10.0, len(ecg_data)/360.0)
    end_idx = int(end_time * 360)
    
    ax2.plot(time[:end_idx], ecg_data[:end_idx], 'b-', linewidth=1.5, label='ECG сигнал')
    
    # Отмечаем пики на увеличенном фрагменте
    for peak in peaks:
        if peak < end_idx:
            ax2.axvline(x=peak/360.0, color='r', alpha=0.5, linewidth=1.5)
            ax2.plot(peak/360.0, ecg_data[peak], 'ro', markersize=4)
    
    ax2.set_xlabel('Время (сек)')
    ax2.set_ylabel('Амплитуда (мВ)')
    ax2.set_title('Увеличенный фрагмент ЭКГ-сигнала (первые 10 секунд)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Дополнительно: статистика по пикам
    if len(peaks) > 1:
        intervals = np.diff(peaks) / 360.0  # интервалы между пиками в секундах
        print(f"\nСтатистика интервалов между пиками (RR-интервалы):")
        print(f"Средний RR-интервал: {np.mean(intervals):.3f} сек")
        print(f"Стандартное отклонение RR: {np.std(intervals):.3f} сек")
        print(f"Минимальный RR-интервал: {np.min(intervals):.3f} сек")
        print(f"Максимальный RR-интервал: {np.max(intervals):.3f} сек")
        print(f"Средняя ЧСС: {60.0/np.mean(intervals):.1f} BPM")
    
except FileNotFoundError as e:
    print(f"Ошибка: файл не найден - {e}")
except Exception as e:
    print(f"Ошибка при обработке данных: {e}")