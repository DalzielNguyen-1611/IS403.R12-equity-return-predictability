import psutil
import time

procs = [p for p in psutil.process_iter(['pid', 'name']) if p.info['name'] in ['ai.exe', 'PresentMonService.exe']]
for p in procs:
    try: p.cpu_percent(None)
    except: pass

time.sleep(1.5)

for p in procs:
    try:
        cpu = p.cpu_percent(None)
        print(f"{p.info['name']} (PID {p.info['pid']}): {cpu:.1f}% CPU")
    except: pass
