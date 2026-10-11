import psutil
import time

# Warm-up cpu_percent
for p in psutil.process_iter(['pid', 'name']):
    try: p.cpu_percent(None)
    except: pass

time.sleep(1.5)

procs = []
for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
    try:
        cpu = p.cpu_percent(None)
        if cpu > 1.0:
            ram = p.info['memory_info'].rss / (1024 * 1024)
            procs.append((p.info['pid'], p.info['name'], cpu, ram))
    except: pass

procs.sort(key=lambda x: x[2], reverse=True)
print(f"Total CPU: {psutil.cpu_percent()}%")
print(f"{'PID':8s} {'Name':25s} {'CPU%':8s} {'RAM(MB)':8s}")
for pid, name, cpu, ram in procs[:15]:
    print(f"{pid:8d} {name:25s} {cpu:8.1f}% {ram:8.1f}")
