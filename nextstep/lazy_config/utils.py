import os
import signal
import socket
import subprocess
import time
from functools import wraps
from shutil import copytree

import debugpy
from pyinstrument import Profiler
from viztracer import VizTracer

from nextstep.utils.loguru import logger
from nextstep.utils.timer import timestamp_str


def wait_for_remote_attach():
    if debugpy.is_client_connected():
        return
    ip = socket.gethostbyname(socket.getfqdn(socket.gethostname()))
    port = 5678
    logger.critical(f"(ip, port): ({ip}, {port}) is waiting for remote attach...")
    logger.info(f"Please run `Python Debugger: Remote Attach` and enter this ip address to attach.")

    debugpy.listen(address=(ip, port))
    debugpy.wait_for_client()

    logger.success(f"Remote attach success!")


profiler_active = True


def cleanup_profiler(profiler: Profiler, viz_tracer: VizTracer | None, timestamp: str, dir: str):
    """Stop and save profiler data."""
    global profiler_active
    if profiler_active:
        try:
            # Stop the pyinstrument Profiler
            if profiler is not None:
                profiler.stop()
                # Save the pyinstrument profile result
                pyinstrument_filename = f"profiler_pyinstrument_{timestamp}_{os.getpid()}.html"
                with open(os.path.join(dir, pyinstrument_filename), "w") as f:
                    f.write(profiler.output_html())
                logger.info(f"Pyinstrument profiler output saved to {os.path.join(dir, pyinstrument_filename)}.")

            # Stop the viztracer
            if viz_tracer is not None:
                viz_tracer.stop()
                viztracer_filename = f"profiler_viztracer_{timestamp}_{os.getpid()}.json"
                viz_tracer.save(os.path.join(dir, viztracer_filename))
                logger.info(f"VizTracer profiler output saved to {os.path.join(dir, viztracer_filename)}.")
            profiler_active = False
        except Exception as e:
            logger.warning(f"Error when saving profiler output: {e}")


def auto_profiler(dir="./", seconds=None, use_profiler=False, use_viz_tracer=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            timestamp = timestamp_str()

            # Start the pyinstrument Profiler
            profiler = None
            if use_profiler:
                profiler = Profiler()
                profiler.start()

            viz_tracer = None
            if use_viz_tracer:
                viz_tracer = VizTracer()
                viz_tracer.start()

            def signal_handler(signum, frame):
                logger.warning("Signal received, stopping profiler...")
                cleanup_profiler(profiler, viz_tracer, timestamp, dir)
                raise SystemExit(f"Terminated due to signal {signum}")

            signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C - KeyboardInterrupt
            signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signal

            if seconds is not None:
                signal.signal(signal.SIGALRM, signal_handler)
                signal.alarm(seconds)

            result = None
            try:
                result = func(*args, **kwargs)
            except SystemExit as e:
                logger.info(str(e))
                raise
            finally:
                if seconds is not None:
                    signal.alarm(0)

                cleanup_profiler(profiler, viz_tracer, timestamp, dir)
            return result

        return wrapper

    return decorator


def copy_codebase(output_dir):
    start_time = time.time()

    def get_git_ignored_files():
        """Get list of files ignored by git"""
        try:
            # Get the git root directory
            git_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], universal_newlines=True).strip()

            # Get list of ignored files
            status_output = subprocess.check_output(
                ["git", "status", "--ignored", "--porcelain"], universal_newlines=True, cwd=git_root
            ).splitlines()

            # Parse the output to get ignored files
            ignored_files = set()
            for line in status_output:
                if line.startswith("!!"):  # !! prefix indicates ignored files
                    # Remove the '!! ' prefix and convert to relative path
                    ignored_file = line[3:]
                    ignored_files.add(ignored_file)

            # .git is ignored by default
            ignored_files.add(".git/")

            return git_root, ignored_files
        except subprocess.CalledProcessError as e:
            logger.error(f"Error {str(e)}:\nNot a git repository or git not installed")
            return None, None

    """Custom ignore function based on git status --ignored"""
    git_root, ignored_files = get_git_ignored_files()

    def git_ignore_filter(src, names):
        if not git_root:
            return set()

        ignore_set = set()
        for name in names:
            # Get relative path from git root
            full_path = os.path.join(src, name)
            rel_path = os.path.relpath(full_path, git_root)
            rel_path = rel_path.replace("\\", "/")  # use forward slash

            # check full path
            # BUG: Don't check directory name, if parent directory has ignored files named the same, it will be ignored
            if rel_path in ignored_files:
                ignore_set.add(name)
                continue

            # check parent directory
            current_path = rel_path
            while current_path:
                if current_path in ignored_files or current_path + "/" in ignored_files:
                    ignore_set.add(name)
                    break
                current_path = os.path.dirname(current_path)

        return ignore_set

    new_code_path = os.path.join(output_dir, "codebase")

    if os.path.exists(new_code_path):
        logger.error(f"Error. {new_code_path} already exists.")
        return -1

    logger.info(f"Copying codebase to {new_code_path}")

    # Get the git root directory
    try:
        current_code_path = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], universal_newlines=True).strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Error {str(e)}:\nNot a git repository or git not installed")
        return -1

    try:
        copytree(current_code_path, new_code_path, ignore=git_ignore_filter)
        logger.info(f"Done copying code. Time taken: {time.time() - start_time:.2f} seconds")
        return 1
    except Exception as e:
        logger.error(f"Error copying codebase: {str(e)}. Time taken: {time.time() - start_time:.2f} seconds")
        return -1


def export_requirements(output_dir, skip_packages=None):
    """
    Export current environment dependencies to requirements.txt file

    Args:
        output_path (str): Output path for requirements file
        skip_packages (list): List of package names to skip
    """
    start_time = time.time()
    requirements_path = os.path.join(output_dir, "requirements.txt")
    logger.info(f"Exporting requirements to {requirements_path}")
    try:
        # Run pip freeze to get all dependencies
        result = subprocess.run(["pip", "freeze"], capture_output=True, text=True, check=True)
        requirements = result.stdout.strip().split("\n")

        # Filter out skipped packages if any
        if skip_packages:
            skip_packages = [pkg.lower() for pkg in skip_packages]
            requirements = [
                req for req in requirements if not any(req.lower().startswith(skip + "==") for skip in skip_packages)
            ]

        # Write to file
        with open(requirements_path, "w", encoding="utf-8") as f:
            f.write("\n".join(requirements))

        logger.info(f"Done exporting requirements. Time taken: {time.time() - start_time:.2f} seconds")
        return 1
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running pip freeze: {e}")
        return -1
    except IOError as e:
        logger.error(f"Error writing to file: {e}")
        return -1
