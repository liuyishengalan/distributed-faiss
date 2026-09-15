#!/usr/bin/env python3

"""Run repeated localhost SIFT1M Flat-L2 experiments with a fixed CPU budget."""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("/home/yslalan/project/dataset/SIFT1M")
    )
    parser.add_argument("--servers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--total-omp-threads", type=int, default=8)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--add-batch-size", type=int, default=10_000)
    parser.add_argument("--query-batch-size", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("results/sift1m_local_matrix"))
    return parser.parse_args()


def summarize(runs):
    summary = []
    for server_count in sorted({run["num_servers"] for run in runs}):
        group = [run for run in runs if run["num_servers"] == server_count]
        summary.append(
            {
                "num_servers": server_count,
                "repeats": len(group),
                "omp_threads_per_server": group[0]["omp_threads_per_server"],
                "total_omp_threads": server_count * group[0]["omp_threads_per_server"],
                "recall_at_1": [run["recall_at_1"] for run in group],
                "recall_at_k": [run["recall_at_k"] for run in group],
                "search_seconds": [run["search_seconds"] for run in group],
                "median_search_seconds": statistics.median(run["search_seconds"] for run in group),
                "median_qps": statistics.median(run["qps"] for run in group),
                "median_build_seconds": statistics.median(run["build_seconds"] for run in group),
                "peak_process_rss_mb": [run["peak_process_rss_mb"] for run in group],
                "all_passed": all(run["passed"] for run in group),
            }
        )
    return summary


def main():
    args = parse_args()
    if args.repeats < 1 or args.total_omp_threads < 1:
        raise ValueError("--repeats and --total-omp-threads must be positive")
    if any(count < 1 for count in args.servers):
        raise ValueError("--servers values must be positive")
    if len(set(args.servers)) != len(args.servers):
        raise ValueError("--servers values must be unique")
    if any(args.total_omp_threads % count for count in args.servers):
        raise ValueError("--total-omp-threads must be divisible by every server count")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = Path(__file__).with_name("benchmark_sift1m.py")
    runs = []
    for server_count in args.servers:
        per_server_threads = args.total_omp_threads // server_count
        for repeat in range(1, args.repeats + 1):
            output_path = args.output_dir / f"flat_l2_{server_count}servers_run{repeat}.json"
            command = [
                sys.executable,
                str(benchmark),
                "--dataset-dir",
                str(args.dataset_dir),
                "--num-servers",
                str(server_count),
                "--k",
                str(args.k),
                "--add-batch-size",
                str(args.add_batch_size),
                "--query-batch-size",
                str(args.query_batch_size),
                "--omp-threads",
                str(per_server_threads),
                "--output",
                str(output_path),
                "--log-level",
                "ERROR",
            ]
            print(
                f"Starting {server_count} server(s), run {repeat}/{args.repeats}, "
                f"{per_server_threads} OpenMP thread(s)/server",
                flush=True,
            )
            started = time.monotonic()
            completed = subprocess.run(command, text=True, capture_output=True)
            if completed.returncode != 0:
                print(completed.stdout, file=sys.stderr)
                print(completed.stderr, file=sys.stderr)
                raise RuntimeError(f"Benchmark failed: {' '.join(command)}")
            result = json.loads(output_path.read_text())
            if not result["passed"]:
                raise RuntimeError(f"Benchmark validation failed: {output_path}")
            result["repeat"] = repeat
            result["wall_seconds"] = time.monotonic() - started
            result["command"] = command
            runs.append(result)
            print(
                f"Completed: search={result['search_seconds']:.2f}s, "
                f"QPS={result['qps']:.2f}, Recall@{args.k}={result['recall_at_k']:.5f}",
                flush=True,
            )

    summary = {
        "workload": "SIFT1M independent-query search; not all-kNN graph construction",
        "execution": "localhost server threads in one process; CPU-only diagnostic",
        "dataset_dir": str(args.dataset_dir.resolve()),
        "k": args.k,
        "query_batch_size": args.query_batch_size,
        "add_batch_size": args.add_batch_size,
        "total_omp_threads": args.total_omp_threads,
        "configurations": summarize(runs),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
