#!/usr/bin/env python3

"""Run repeated SIFT1M Flat-L2 tests with independent localhost server processes."""

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import tempfile
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
    parser.add_argument("--max-base", type=int, default=None, help="Quick test only")
    parser.add_argument("--max-queries", type=int, default=None, help="Quick test only")
    parser.add_argument("--output-dir", type=Path, default=Path("results/sift1m_external_matrix"))
    return parser.parse_args()


def reserve_free_ports(count):
    sockets = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def wait_for_servers(children, ports, timeout_seconds=30):
    deadline = time.monotonic() + timeout_seconds
    for child, port in zip(children, ports):
        while True:
            if child.poll() is not None:
                raise RuntimeError(f"Server process {child.pid} exited before readiness")
            try:
                with socket.create_connection(("localhost", port), timeout=1):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Server on port {port} did not become reachable")
                time.sleep(0.1)


def read_vm_hwm_mb(pid):
    """Read a Linux process's peak resident set size while it is still alive."""
    status = Path(f"/proc/{pid}/status").read_text()
    for line in status.splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError(f"VmHWM not available for process {pid}")


def summarize(runs):
    configurations = []
    for server_count in sorted({run["num_servers"] for run in runs}):
        group = [run for run in runs if run["num_servers"] == server_count]
        configurations.append(
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
                "median_client_peak_rss_mb": statistics.median(
                    run["peak_process_rss_mb"] for run in group
                ),
                "median_sum_server_vm_hwm_mb": statistics.median(
                    run["sum_server_vm_hwm_mb"] for run in group
                ),
                "all_passed": all(run["passed"] for run in group),
            }
        )
    return configurations


def run_once(args, server_count, repeat, per_server_threads, output_path):
    script_dir = Path(__file__).resolve().parent
    ports = reserve_free_ports(server_count)
    children = []
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="distributed-faiss-process-matrix-") as temp_dir:
        temp_path = Path(temp_dir)
        storage_dir = temp_path / "indexes"
        discovery = temp_path / "servers.txt"
        discovery.write_text(
            str(server_count) + "\n" + "".join(f"localhost,{port}\n" for port in ports)
        )
        try:
            for rank, port in enumerate(ports):
                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = str(per_server_threads)
                env["OPENBLAS_NUM_THREADS"] = "1"
                log_path = temp_path / f"server_{rank}.log"
                with log_path.open("w") as log_file:
                    children.append(
                        subprocess.Popen(
                            [
                                sys.executable,
                                str(script_dir / "serve_index.py"),
                                "--rank",
                                str(rank),
                                "--port",
                                str(port),
                                "--storage-dir",
                                str(storage_dir),
                            ],
                            env=env,
                            stdout=log_file,
                            stderr=subprocess.STDOUT,
                        )
                    )
            wait_for_servers(children, ports)
            command = [
                sys.executable,
                str(script_dir / "benchmark_sift1m.py"),
                "--dataset-dir",
                str(args.dataset_dir),
                "--discovery-config",
                str(discovery),
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
            if args.max_base is not None:
                command.extend(["--max-base", str(args.max_base)])
            if args.max_queries is not None:
                command.extend(["--max-queries", str(args.max_queries)])
            client_env = os.environ.copy()
            client_env["OMP_NUM_THREADS"] = "1"
            client_env["OPENBLAS_NUM_THREADS"] = "1"
            completed = subprocess.run(command, env=client_env, text=True, capture_output=True)
            if completed.returncode:
                print(completed.stdout, file=sys.stderr)
                print(completed.stderr, file=sys.stderr)
                for rank in range(server_count):
                    log = (temp_path / f"server_{rank}.log").read_text()
                    print(f"server {rank} log tail:\n{log[-20000:]}", file=sys.stderr)
                raise RuntimeError(f"Benchmark failed for {server_count} servers, run {repeat}")
            if any(child.poll() is not None for child in children):
                raise RuntimeError("A server process exited during the benchmark")
            run = json.loads(output_path.read_text())
            if not run["passed"] or run["num_servers"] != server_count:
                raise RuntimeError(f"Benchmark validation failed: {output_path}")
            run["repeat"] = repeat
            run["wall_seconds"] = time.monotonic() - started
            run["server_pids"] = [child.pid for child in children]
            run["server_vm_hwm_mb"] = [read_vm_hwm_mb(child.pid) for child in children]
            run["sum_server_vm_hwm_mb"] = sum(run["server_vm_hwm_mb"])
            run["command"] = command
            output_path.write_text(json.dumps(run, indent=2) + "\n")
            return run
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for child in children:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def main():
    args = parse_args()
    if args.repeats < 1 or args.total_omp_threads < 1:
        raise ValueError("--repeats and --total-omp-threads must be positive")
    if any(count < 1 for count in args.servers) or len(set(args.servers)) != len(args.servers):
        raise ValueError("--servers must contain distinct positive counts")
    if any(args.total_omp_threads % count for count in args.servers):
        raise ValueError("--total-omp-threads must be divisible by every server count")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for server_count in args.servers:
        per_server_threads = args.total_omp_threads // server_count
        for repeat in range(1, args.repeats + 1):
            output_path = args.output_dir / f"flat_l2_{server_count}servers_run{repeat}.json"
            if output_path.exists():
                raise FileExistsError(f"Refusing to overwrite an existing result: {output_path}")
            print(
                f"Starting {server_count} process(es), run {repeat}/{args.repeats}, "
                f"{per_server_threads} OpenMP thread(s)/server",
                flush=True,
            )
            run = run_once(args, server_count, repeat, per_server_threads, output_path)
            runs.append(run)
            print(
                f"Completed: search={run['search_seconds']:.2f}s, "
                f"QPS={run['qps']:.2f}, Recall@{args.k}={run['recall_at_k']:.5f}, "
                f"sum server peak RSS={run['sum_server_vm_hwm_mb']:.1f} MB",
                flush=True,
            )

    summary = {
        "workload": "SIFT1M independent-query search; not all-kNN graph construction",
        "execution": "independent localhost server processes; CPU-only diagnostic",
        "dataset_dir": str(args.dataset_dir.resolve()),
        "k": args.k,
        "query_batch_size": args.query_batch_size,
        "add_batch_size": args.add_batch_size,
        "total_omp_threads": args.total_omp_threads,
        "client_openmp_threads": 1,
        "memory_note": "VmHWM is per-server peak RSS; its sum is not a simultaneous memory peak",
        "configurations": summarize(runs),
    }
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing summary: {summary_path}")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
