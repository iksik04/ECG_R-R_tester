import wfdb
import numpy as np
from pathlib import Path

def save_atr(peaks, output_path, symbols=None):
    """
    Сохраняет список пиков в формате WFDB .atr.
    Если symbols не указаны, всем пикам присваивается символ 'N'.
    """
    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    peaks_sorted = sorted(peaks)
    peak_array = np.array(peaks_sorted, dtype=np.int32)
    
    if symbols is None:
        symbols = np.array(['N'] * len(peak_array), dtype='U1')
    else:
        symbols = np.array(symbols, dtype='U1')
    
    # Имя записи без расширения
    base_name = output_path.stem
    
    wfdb.wrann(
        record_name=base_name,
        extension='atr',
        sample=peak_array,
        symbol=symbols,
        write_dir=str(output_dir)
    )