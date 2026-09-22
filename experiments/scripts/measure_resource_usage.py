from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "raw" / "resource_measurement"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Python experiment as a child process while measuring "
            "its peak resident set size (RSS)."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "resource_measurement.json",
        help="Path for the resource measurement JSON file.",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Memory sampling interval in seconds. Default: 0.5.",
    )

    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help=(
            "Command to execute after --. Example: "
            "-- python -m experiments.scripts.run_centralized --epochs 1"
        ),
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.interval <= 0:
        raise ValueError("--interval must be greater than zero.")

    if not args.command:
        raise ValueError(
            "No command supplied. Put the experiment command after --."
        )

    if args.command[0] == "--":
        args.command = args.command[1:]

    if not args.command:
        raise ValueError("No command supplied after --.")


def normalize_python_command(command: Sequence[str]) -> list[str]:
    """
    Replace a generic Python executable name with the exact interpreter
    running this resource-monitor process.
    """
    normalized = list(command)

    if not normalized:
        return normalized

    executable_name = Path(normalized[0]).name.lower()

    python_names = {
        "python",
        "python.exe",
        "python3",
        "python3.exe",
    }

    if executable_name in python_names:
        normalized[0] = sys.executable

    return normalized


def process_rss_bytes(process: psutil.Process) -> int:
    """Return the current RSS of a process in bytes."""
    try:
        return process.memory_info().rss
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


def collect_process_tree_rss_bytes(root_process: psutil.Process) -> int:
    """
    Return RSS for the root process plus any currently running children.

    This makes the measurement robust if an experiment later creates child
    processes. The current project normally runs as one Python process.
    """
    total_rss = 0

    processes: list[psutil.Process] = [root_process]

    try:
        processes.extend(root_process.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    seen_pids: set[int] = set()

    for process in processes:
        try:
            if process.pid in seen_pids:
                continue

            seen_pids.add(process.pid)
            total_rss += process_rss_bytes(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return total_rss


def bytes_to_mib(value: int) -> float:
    return value / (1024 * 1024)


def run_and_measure(
    command: Sequence[str],
    interval_seconds: float,
) -> dict:
    """
    Run a child process and continuously sample process-tree RSS.

    Peak memory is reported as the maximum sampled RSS of the root process
    plus any child processes.
    """
    normalized_command = normalize_python_command(command)

    start_time = time.perf_counter()

    completed_process = subprocess.Popen(
        normalized_command,
        cwd=PROJECT_ROOT,
    )

    process = psutil.Process(completed_process.pid)

    peak_rss_bytes = 0
    samples = 0

    current_rss = collect_process_tree_rss_bytes(process)
    peak_rss_bytes = max(peak_rss_bytes, current_rss)
    samples += 1

    while completed_process.poll() is None:
        current_rss = collect_process_tree_rss_bytes(process)
        peak_rss_bytes = max(peak_rss_bytes, current_rss)
        samples += 1

        time.sleep(interval_seconds)

    current_rss = collect_process_tree_rss_bytes(process)
    peak_rss_bytes = max(peak_rss_bytes, current_rss)
    samples += 1

    elapsed_seconds = time.perf_counter() - start_time

    return {
        "status": "completed" if completed_process.returncode == 0 else "failed",
        "command_requested": list(command),
        "command_executed": normalized_command,
        "working_directory": str(PROJECT_ROOT),
        "process_id": completed_process.pid,
        "exit_code": completed_process.returncode,
        "measurement": {
            "metric": "peak_process_tree_rss",
            "peak_rss_bytes": peak_rss_bytes,
            "peak_rss_mib": bytes_to_mib(peak_rss_bytes),
            "sampling_interval_seconds": interval_seconds,
            "samples_collected": samples,
        },
        "tool": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "psutil_version": psutil.__version__,
        },
        "wall_time_seconds": elapsed_seconds,
        "notes": [
            "Peak memory is measured as resident set size (RSS).",
            "RSS is sampled periodically and therefore represents the peak observed sample, not an instantaneous hardware-level maximum.",
            "The measurement includes the launched Python process and any child processes that exist during sampling.",
            "Generic Python executable names are replaced with the exact interpreter running this resource monitor.",
            "This is process memory, not total system RAM usage.",
            "Network traffic, disk cache, and operating-system-wide memory usage are not represented by this metric.",
        ],
    }


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("RESOURCE MEASUREMENT")
    print("=" * 72)
    print(f"Requested command:    {' '.join(args.command)}")

    normalized_command = normalize_python_command(args.command)
    print(f"Executed command:     {' '.join(normalized_command)}")

    print(f"Working directory:    {PROJECT_ROOT}")
    print(f"Sampling interval:    {args.interval:.3f} seconds")
    print(f"Monitor Python:       {sys.executable}")
    print(f"Output:               {output_path}")
    print()

    result = run_and_measure(
        command=args.command,
        interval_seconds=args.interval,
    )

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print("Measurement complete.")
    print(f"Status:               {result['status']}")
    print(f"Exit code:             {result['exit_code']}")
    print(
        "Peak process RSS:     "
        f"{result['measurement']['peak_rss_bytes']:,} bytes"
    )
    print(
        "Peak process RSS:     "
        f"{result['measurement']['peak_rss_mib']:.3f} MiB"
    )
    print(
        "Samples collected:    "
        f"{result['measurement']['samples_collected']}"
    )
    print(
        "Wall time:             "
        f"{result['wall_time_seconds']:.3f} seconds"
    )
    print(f"Result JSON:           {output_path}")

    if result["exit_code"] != 0:
        print(
            "\nThe monitored command failed. "
            "The resource JSON was still written so the failure is recorded.",
            file=sys.stderr,
        )
        return result["exit_code"]

    return 0


if __name__ == "__main__":
    raise SystemExit(main())