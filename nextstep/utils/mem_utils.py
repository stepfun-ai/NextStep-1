import os
import threading
import time

import psutil

from nextstep.utils.loguru import logger


class MemoryMonitor:
    def __init__(self, name):
        self.name = name
        self.pid = os.getpid()
        self.process = psutil.Process(self.pid)

        self.system_memory = psutil.virtual_memory()

    @property
    def GB(self):
        return 1 << 30

    @property
    def rss(self):
        return f"{self.process.memory_info().rss / self.GB:.2f} GB"

    @property
    def vms(self):
        return f"{self.process.memory_info().vms / self.GB:.2f} GB"

    @property
    def total_mem(self):
        return f"{self.system_memory.total / self.GB:.2f} GB"

    @property
    def used_mem(self):
        return f"{self.system_memory.used / self.GB:.2f} GB"

    @property
    def available_mem(self):
        return f"{self.system_memory.available / self.GB:.2f} GB"

    @property
    def free_mem(self):
        return f"{self.system_memory.free / self.GB:.2f} GB"

    @property
    def buffers_cached_mem(self):
        return f"{self.system_memory.buffers + self.system_memory.cached / self.GB:.2f} GB"

    @property
    def percent(self):
        return f"{self.system_memory.percent:.2f}%"

    def log_memory(self, message):
        logger.info(f"[{message}] {self.name} (pid-{self.pid}) --> RSS: {self.rss}, VMS: {self.vms}")
        logger.info(
            f"\n"
            f"[SYSTEM_MEMORY_INFO]\n"
            f"└── Total: {self.total_mem}\n"
            f"    ├── Used: {self.used_mem}\n"
            f"    └── Available: {self.available_mem}\n"
            f"        ├── Free: {self.free_mem}\n"
            f"        └── Buffers&Cached: {self.buffers_cached_mem}\n"
            f"    Memory Usage: {self.percent}"
        )


class PeriodicMemoryMonitor:
    def __init__(self, name, interval=60):
        self.interval = interval
        self.stop_flag = False
        self.memory_monitor = MemoryMonitor(name)
        self.thread = None  # Initialize thread as None

    def start(self):
        if self.interval == -1:
            return
        if self.thread is None:  # Only start if not already running
            self.stop_flag = False
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()

    def stop(self):
        if self.interval == -1:
            return
        if self.thread is not None:  # Only attempt to stop if thread exists
            self.stop_flag = True
            try:
                self.thread.join(timeout=1.0)  # Add timeout to avoid hanging
            except Exception:
                pass  # Ignore any errors during thread cleanup
            self.thread = None

    def _monitor_loop(self):
        while not self.stop_flag:
            try:
                self.memory_monitor.log_memory("PeriodicMemoryMonitor")
                time.sleep(self.interval)
            except Exception:
                break  # Exit loop on any error
