#!/usr/bin/env python3

"""Launch independent localhost server processes for a SIFT1M correctness test."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from benchmark_sift1m import reserve_free_ports, wait_for_external_servers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("/home/yslalan/project/dataset/SIFT1M")
    )
    parser.add_argument("--num-servers", type=int, default=2)
    parser.add_argument("--max-base", type=int, default=10_000)
    parser.add_argument("--max-queries", type=int, default=20)
    parser.add_argument("--add-batch-size", type=int, default=1_000)
    parser.add_argument("--omp-threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.num_servers < 1:
        raise ValueError("--num-servers must be positive")

    script_dir = Path(__file__).resolve().parent
    serve_script = script_dir / "serve_index.py"
    benchmark_script = script_dir / "benchmark_sift1m.py"
    ports = reserve_free_ports(args.num_servers)
    children = []

    with tempfile.TemporaryDirectory(prefix="distributed-faiss-external-smoke-") as temp_dir:
        temp_path = Path(temp_dir)
        storage_dir = temp_path / "indexes"
        discovery_path = temp_path / "servers.txt"
        discovery_path.write_text(
            str(args.num_servers) + "\n" + "".join(f"localhost,{port}\n" for port in ports)
        )
        try:
            for rank, port in enumerate(ports):
                log_path = temp_path / f"server_{rank}.log"
                with log_path.open("w") as log_file:
                    children.append(
                        subprocess.Popen(
                            [
                                sys.executable,
                                str(serve_script),
                                "--rank",
                                str(rank),
                                "--port",
                                str(port),
                                "--storage-dir",
                                str(storage_dir),
                            ],
                            stdout=log_file,
                            stderr=subprocess.STDOUT,
                        )
                    )

            wait_for_external_servers([("localhost", port) for port in ports], 30.0)
            benchmark_command = [
                sys.executable,
                str(benchmark_script),
                "--dataset-dir",
                str(args.dataset_dir),
                "--discovery-config",
                str(discovery_path),
                "--max-base",
                str(args.max_base),
                "--max-queries",
                str(args.max_queries),
                "--add-batch-size",
                str(args.add_batch_size),
                "--omp-threads",
                str(args.omp_threads),
                "--log-level",
                "ERROR",
            ]
            if args.output is not None:
                benchmark_command.extend(["--output", str(args.output)])
            completed = subprocess.run(benchmark_command, text=True, capture_output=True)
            print(completed.stdout)
            if completed.returncode:
                print(completed.stderr, file=sys.stderr)
                for rank in range(args.num_servers):
                    print((temp_path / f"server_{rank}.log").read_text(), file=sys.stderr)
                raise RuntimeError("External-server SIFT1M smoke test failed")
            if any(child.poll() is not None for child in children):
                raise RuntimeError("An external server exited while the benchmark was running")
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


if __name__ == "__main__":
    main()
