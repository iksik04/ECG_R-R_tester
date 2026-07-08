import sys
import time

def print_progress(completed, total, successful, failed, start_time):
    """
    Печатает прогресс-бар на одной строке.
    """
    elapsed = time.time() - start_time
    avg_time = elapsed / completed if completed > 0 else 0
    remaining = (total - completed) * avg_time if avg_time > 0 else 0
    
    bar_width = 40
    filled = int(bar_width * completed / total)
    bar = '█' * filled + '░' * (bar_width - filled)
    
    remaining_str = f"{remaining:.0f}s" if remaining > 0 else "0s"
    
    progress_str = (f"\r[{bar}] {completed}/{total} "
                    f"| OK {successful} ERR {failed} "
                    f"| ~{remaining_str} сек")
    
    sys.stdout.write(progress_str)
    sys.stdout.flush()