import wfdb
import numpy as np
from pathlib import Path

def read_wfdb(record_path):
    """
    Читает WFDB запись (файлы .hea, .dat) и возвращает сигнал (первый канал) и частоту дискретизации.
    """
    record_path = Path(record_path)
    try:
        record = wfdb.rdrecord(str(record_path), physical=True)
    except Exception as e:
        raise RuntimeError(f"Ошибка чтения WFDB записи {record_path}: {e}")
    
    if record.p_signal is None or record.p_signal.size == 0:
        raise ValueError("Не удалось прочитать сигналы из записи")
    
    signal = record.p_signal[:, 0]  # берём первый канал
    fs = record.fs
    return signal, fs