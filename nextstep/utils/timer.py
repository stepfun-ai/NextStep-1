import functools
import signal
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Literal

import pytz
import schedule

from nextstep.utils.loguru import logger, setup_logger


def timestamp_str(*, time_value=None, zone="Asia/Shanghai") -> str:
    """format given timestamp, if no timestamp is given, return a call time string"""
    if time_value is None:
        time_value = datetime.now(pytz.timezone(zone))
    return time_value.strftime("%Y-%m-%d_%H-%M-%S")


class TimerData:
    def __init__(self, alpha=0.8):
        self.times = []
        self.ema = None
        self.alpha = alpha

    @property
    def avg(self):
        return sum(self.times) / len(self.times) if self.times else 0

    @property
    def moving_avg(self):
        return self.ema if self.ema is not None else self.avg

    @property
    def value(self):
        return self.times[-1] if self.times else 0

    def acc_value(self, acc_steps: int):
        return sum(self.times[-acc_steps:]) if self.times else 0

    @property
    def total(self):
        return sum(self.times)

    def add_time(self, time):
        self.times.append(time)
        if self.ema is None:
            self.ema = time
        else:
            self.ema = self.alpha * time + (1 - self.alpha) * self.ema


class TimerManager:
    def __init__(self):
        self.timers = defaultdict(TimerData)
        self.start_times = {}

    def beat(self, key: str, action: Literal["start", "end"]):
        current_time = datetime.now()
        if action == "start":
            self.start_times[key] = current_time
        elif action == "end":
            if key in self.start_times:
                elapsed_time = (current_time - self.start_times[key]).total_seconds()
                self.timers[key].add_time(elapsed_time)
            else:
                logger.warning(f"'end' beat called for {key} without a corresponding 'start'.")

    def __getattr__(self, key):
        return self.timers[key]

    def reset(self, key: str = None):
        if key is None:
            self.timers.clear()
            self.start_times.clear()
        else:
            self.timers[key].times = []
            self.timers[key].ema = None
            self.start_times.pop(key, None)


def format_time(time: int | float | timedelta):
    if isinstance(time, timedelta):
        total_seconds = int(time.total_seconds())
    elif isinstance(time, float):
        total_seconds = round(time)
    else:
        total_seconds = time

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    # Format the string
    formatted_string = f"{hours}:{minutes:02}:{seconds:02}"
    return formatted_string


class TimeoutError(Exception):
    pass


def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")

            original_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, original_handler)

        return wrapper

    return decorator


class ShellCommandTimer:
    def __init__(self):
        self.scheduler = schedule.Scheduler()
        setup_logger(_logger=logger, save_dir="./timer_logs", filename=f"{timestamp_str()}_timer.log")
        self.logger = logger

    def execute_command(self, command):
        """
        Execute shell command and log results
        :param command: shell command string
        """
        try:
            self.logger.info(f"Executing command: {command}")
            result = subprocess.run(command, shell=True, capture_output=True, text=True, encoding="utf-8")

            if result.returncode == 0:
                self.logger.info(f"Command executed successfully:\n{result.stdout}")
            else:
                self.logger.error(f"Command execution failed:\n{result.stderr}")

            return result
        except Exception as e:
            self.logger.error(f"Error executing command: {str(e)}")
            return None

    def has_pending_jobs(self):
        """
        Check if there are any jobs in the scheduler
        :return: Boolean indicating if there are any jobs
        """
        return len(self.scheduler.jobs) > 0

    def schedule_command(self, target_time, command):
        """
        Schedule a shell command at specific time
        :param target_time: time string in "HH:MM" format, e.g., "14:30"
        :param command: shell command to execute
        """
        self.scheduler.every().day.at(target_time).do(lambda: self.execute_command(command))
        self.logger.info(f"Scheduled command '{command}' to run daily at {target_time}")

    def schedule_interval_command(self, interval_seconds, command):
        """
        Schedule a shell command to run at fixed intervals
        :param interval_seconds: interval in seconds
        :param command: shell command to execute
        """
        self.scheduler.every(interval_seconds).seconds.do(lambda: self.execute_command(command))
        self.logger.info(f"Scheduled command '{command}' to run every {interval_seconds} seconds")

    def schedule_hours_later(self, hours, command):
        """
        Schedule a command to run specified hours from now
        :param hours: number of hours to delay execution
        :param command: shell command to execute
        """
        # Calculate target time
        target_time = datetime.now() + timedelta(hours=hours)
        target_time_str = target_time.strftime("%H:%M")

        # Check if target time is tomorrow
        if target_time.date() > datetime.now().date():
            self.logger.info(f"Target time will be tomorrow at {target_time_str}")
        else:
            self.logger.info(f"Target time will be today at {target_time_str}")

        def one_time_task():
            self.execute_command(command)
            return schedule.CancelJob  # Cancel task after execution

        self.scheduler.every().day.at(target_time_str).do(one_time_task)
        self.logger.info(f"Scheduled command '{command}' to run at {target_time.strftime('%Y-%m-%d %H:%M')}")

    def run(self, auto_stop=True):
        """
        Start the timer
        :param auto_stop: If True, stops when no jobs are pending
        """
        self.logger.info("Timer started")

        if not self.has_pending_jobs():
            self.logger.warning("No jobs scheduled. Timer stopping.")
            return

        try:
            while True:
                self.scheduler.run_pending()

                # Check if we should auto-stop
                if auto_stop and not self.has_pending_jobs():
                    self.logger.info("All jobs completed. Timer stopping.")
                    break

                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Timer stopped by user")
        except Exception as e:
            self.logger.error(f"Error occurred: {str(e)}")


def example_usage():
    # Example 1: One-time task - will auto-stop after completion
    timer = ShellCommandTimer()
    timer.schedule_hours_later(0.02, "echo 'Task completed!'")  # Runs after 1 minutes
    timer.run(auto_stop=True)  # Will stop after the task is done

    # Example 2: Multiple tasks including repeating ones
    timer = ShellCommandTimer()
    timer.schedule_interval_command(5, "echo 'Repeating task'")  # Runs every 5 seconds
    timer.schedule_hours_later(0.02, "echo 'One-time task' after 0.02 hours")  # One-time task
    timer.schedule_command("14:30", "ls -l")
    timer.run(auto_stop=False)  # Won't auto-stop due to repeating task


if __name__ == "__main__":
    example_usage()
