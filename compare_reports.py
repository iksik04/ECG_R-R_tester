import re
from pathlib import Path
import argparse

def parse_report(file_path):
    """Парсит отчет и возвращает словарь с данными"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    results = {}
    
    # Количество записей
    match = re.search(r'Всего записей:\s*(\d+)', content)
    if match:
        results['total_records'] = int(match.group(1))
    
    # Общие метрики
    match = re.search(r'Общая чувствительность QRS \(Se\).*?:\s*([\d.]+)', content)
    if match:
        results['overall_sensitivity'] = float(match.group(1))
    
    match = re.search(r'Общая специфичность QRS \(\+P\).*?:\s*([\d.]+)', content)
    if match:
        results['overall_specificity'] = float(match.group(1))
    
    match = re.search(r'Общее PPV.*?:\s*([\d.]+)', content)
    if match:
        results['overall_ppv'] = float(match.group(1))
    
    # TP, FP, FN, TN
    match = re.search(r'True Positives \(TP\):\s*(\d+)', content)
    if match:
        results['tp'] = int(match.group(1))
    
    match = re.search(r'True Negatives \(TN\):\s*(\d+)', content)
    if match:
        results['tn'] = int(match.group(1))
    
    match = re.search(r'False Positives \(FP\):\s*(\d+)', content)
    if match:
        results['fp'] = int(match.group(1))
    
    match = re.search(r'False Negatives \(FN\):\s*(\d+)', content)
    if match:
        results['fn'] = int(match.group(1))
    
    # Статистика пиков
    match = re.search(r'Всего истинных пиков:\s*(\d+)', content)
    if match:
        results['true_peaks'] = int(match.group(1))
    
    match = re.search(r'Всего обнаруженных пиков:\s*(\d+)', content)
    if match:
        results['detected_peaks'] = int(match.group(1))
    
    # Список записей и их метрики
    records = []
    record_pattern = r'(\w+):\s*Чувствительность:\s*([\d.]+)\s*Специфичность:\s*([\d.]+)\s*PPV:\s*([\d.]+)\s*TP:\s*(\d+),\s*FP:\s*(\d+),\s*FN:\s*(\d+),\s*TN:\s*(\d+)'
    for match in re.finditer(record_pattern, content):
        records.append({
            'name': match.group(1),
            'sensitivity': float(match.group(2)),
            'specificity': float(match.group(3)),
            'ppv': float(match.group(4)),
            'tp': int(match.group(5)),
            'fp': int(match.group(6)),
            'fn': int(match.group(7)),
            'tn': int(match.group(8))
        })
    results['records'] = records
    
    return results

def compare_reports(file1, file2):
    """Сравнивает два отчета"""
    report1 = parse_report(file1)
    report2 = parse_report(file2)
    
    print("=" * 80)
    print("СРАВНЕНИЕ ОТЧЕТОВ")
    print("=" * 80)
    print(f"Отчет 1: {file1}")
    print(f"Отчет 2: {file2}")
    print()
    
    # Сравнение общих метрик
    print("ОБЩИЕ МЕТРИКИ:")
    print("-" * 60)
    
    metrics = [
        ('total_records', 'Всего записей', '{}'),
        ('tp', 'TP', '{}'),
        ('fp', 'FP', '{}'),
        ('fn', 'FN', '{}'),
        ('tn', 'TN', '{}'),
        ('true_peaks', 'Истинные пики', '{}'),
        ('detected_peaks', 'Обнаруженные пики', '{}'),
        ('overall_sensitivity', 'Чувствительность', '{:.4f}'),
        ('overall_specificity', 'Специфичность', '{:.4f}'),
        ('overall_ppv', 'PPV', '{:.4f}')
    ]
    
    for key, name, fmt in metrics:
        val1 = report1.get(key, 'N/A')
        val2 = report2.get(key, 'N/A')
        if val1 != 'N/A' and val2 != 'N/A':
            diff = val1 - val2 if isinstance(val1, (int, float)) else 'N/A'
            if isinstance(diff, (int, float)):
                diff_str = f"{diff:+.4f}" if isinstance(diff, float) else f"{diff:+d}"
            else:
                diff_str = str(diff)
            print(f"{name:20}: {fmt.format(val1)} -> {fmt.format(val2)} (разница: {diff_str})")
    
    print()
    print("СПИСКИ ЗАПИСЕЙ:")
    print("-" * 60)
    
    records1 = {r['name']: r for r in report1.get('records', [])}
    records2 = {r['name']: r for r in report2.get('records', [])}
    
    names1 = set(records1.keys())
    names2 = set(records2.keys())
    
    only_in_1 = names1 - names2
    only_in_2 = names2 - names1
    common = names1 & names2
    
    if only_in_1:
        print(f"Только в отчете 1: {', '.join(sorted(only_in_1))}")
    if only_in_2:
        print(f"Только в отчете 2: {', '.join(sorted(only_in_2))}")
    
    print()
    print("СРАВНЕНИЕ ПОЗАПИСНОЙ СТАТИСТИКИ (общие записи):")
    print("-" * 60)
    
    for name in sorted(common):
        r1 = records1[name]
        r2 = records2[name]
        print(f"\n{name}:")
        print(f"  Чувствительность: {r1['sensitivity']:.4f} -> {r2['sensitivity']:.4f} (разница: {r2['sensitivity'] - r1['sensitivity']:+.4f})")
        print(f"  Специфичность:    {r1['specificity']:.4f} -> {r2['specificity']:.4f} (разница: {r2['specificity'] - r1['specificity']:+.4f})")
        print(f"  PPV:              {r1['ppv']:.4f} -> {r2['ppv']:.4f} (разница: {r2['ppv'] - r1['ppv']:+.4f})")
        print(f"  TP: {r1['tp']} -> {r2['tp']} (разница: {r2['tp'] - r1['tp']:+d})")
        print(f"  FP: {r1['fp']} -> {r2['fp']} (разница: {r2['fp'] - r1['fp']:+d})")
        print(f"  FN: {r1['fn']} -> {r2['fn']} (разница: {r2['fn'] - r1['fn']:+d})")
        print(f"  TN: {r1['tn']} -> {r2['tn']} (разница: {r2['tn'] - r1['tn']:+d})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Сравнение отчетов валидации')
    parser.add_argument('report1', help='Путь к первому отчету')
    parser.add_argument('report2', help='Путь ко второму отчету')
    args = parser.parse_args()
    
    compare_reports(args.report1, args.report2)