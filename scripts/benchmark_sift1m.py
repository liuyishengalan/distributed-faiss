#!/usr/bin/env python3

"""Benchmark a localhost distributed-faiss Flat-L2 index on SIFT1M."""

import argparse
import json
import logging
import math
import socket
import tempfile
import threading
import time
from pathlib import Path

import faiss
import numpy as np

from distributed_faiss.client import IndexClient
from distributed_faiss.index_cfg import IndexCfg
from distributed_faiss.index_state import IndexState
from distributed_faiss.server import IndexServer

DEFAULT_DATASET_DIR = Path("/home/yslalan/project/dataset/SIFT1M")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--num-servers", type=int, default=2)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--add-batch-size", type=int, default=10_000)
    parser.add_argument("--query-batch-size", type=int, default=100)
    parser.add_argument(
        "--max-base",
        type=int,
        default=None,
        help="Use a prefix of the base vectors. Intended for smoke tests.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Use a prefix of the query vectors.",
    )
    parser.add_argument("--omp-threads", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args()


def mmap_vecs(path, value_dtype):
    """Memory-map an fvecs/ivecs file and return its payload and record width."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size < 4:
        raise ValueError(f"Vector file is empty or truncated: {path}")

    raw = np.memmap(path, dtype=np.int32, mode="r")
    width = int(raw[0])
    if width <= 0:
        raise ValueError(f"Invalid record width {width} in {path}")
    record_width = width + 1
    if raw.size % record_width:
        raise ValueError(f"File size is not a multiple of its record width: {path}")

    records = raw.reshape(-1, record_width)
    if not np.all(records[:, 0] == width):
        raise ValueError(f"Inconsistent record headers in {path}")
    payload = records[:, 1:]
    if value_dtype == np.float32:
        payload = payload.view(np.float32)
    elif value_dtype != np.int32:
        raise TypeError(f"Unsupported vector dtype: {value_dtype}")
    return payload, width


def reserve_free_ports(count):
    ports = []
    sockets = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            ports.append(sock.getsockname()[1])
            sockets.append(sock)
    finally:
        for sock in sockets:
            sock.close()
    return ports


def wait_until_trained(client, index_id, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if client.get_state(index_id) == IndexState.TRAINED:
            return
        time.sleep(0.05)
    raise TimeoutError(f"Index {index_id!r} did not become trained")


def recall_at_k(actual, expected, k):
    if actual.shape != (expected.shape[0], k):
        raise ValueError(
            f"Actual result shape {actual.shape} does not match ({expected.shape[0]}, {k})"
        )
    hits = sum(
        len(set(actual_row).intersection(expected_row[:k]))
        for actual_row, expected_row in zip(actual, expected)
    )
    return hits / (expected.shape[0] * k)


def exact_reference(base, queries, k):
    index = faiss.IndexFlatL2(base.shape[1])
    for start in range(0, len(base), 100_000):
        index.add(np.ascontiguousarray(base[start : start + 100_000], dtype=np.float32))
    _, ids = index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
    return ids


def validate_args(args, base_count, query_count, groundtruth_width):
    if args.num_servers < 1:
        raise ValueError("--num-servers must be positive")
    if args.k < 1:
        raise ValueError("--k must be positive")
    if args.add_batch_size < 1 or args.query_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    if args.max_base is not None and not 1 <= args.max_base <= base_count:
        raise ValueError(f"--max-base must be between 1 and {base_count}")
    if args.max_queries is not None and not 1 <= args.max_queries <= query_count:
        raise ValueError(f"--max-queries must be between 1 and {query_count}")
    selected_base_count = args.max_base or base_count
    if args.k > selected_base_count:
        raise ValueError(f"--k={args.k} exceeds the selected base size {selected_base_count}")
    if args.max_base is None and args.k > groundtruth_width:
        raise ValueError(f"--k={args.k} exceeds the {groundtruth_width} neighbors in ground truth")


def main():
    args = parse_args()
    log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid --log-level: {args.log_level}")
    logging.getLogger().setLevel(log_level)

    load_start = time.perf_counter()
    base, dimension = mmap_vecs(args.dataset_dir / "base.fvecs", np.float32)
    queries, query_dimension = mmap_vecs(args.dataset_dir / "query.fvecs", np.float32)
    groundtruth, groundtruth_width = mmap_vecs(args.dataset_dir / "groundtruth.ivecs", np.int32)
    if query_dimension != dimension:
        raise ValueError(f"Base dimension {dimension} != query dimension {query_dimension}")
    if len(queries) != len(groundtruth):
        raise ValueError(f"Query count {len(queries)} != ground-truth count {len(groundtruth)}")
    validate_args(args, len(base), len(queries), groundtruth_width)

    full_base_count = len(base)
    base_count = args.max_base or full_base_count
    query_count = args.max_queries or len(queries)
    base = base[:base_count]
    queries = np.ascontiguousarray(queries[:query_count], dtype=np.float32)
    if base_count == full_base_count:
        expected_ids = np.asarray(groundtruth[:query_count, : args.k], dtype=np.int64)
        reference_source = "sift1m_groundtruth"
    else:
        expected_ids = exact_reference(base, queries, args.k)
        reference_source = "faiss_index_flat_l2_subset"
    load_seconds = time.perf_counter() - load_start

    num_add_batches = math.ceil(base_count / args.add_batch_size)
    if num_add_batches < args.num_servers:
        raise ValueError(
            "Each server must receive at least one batch; lower --num-servers or "
            "--add-batch-size"
        )

    ports = reserve_free_ports(args.num_servers)
    servers = []
    threads = []
    client = None

    with tempfile.TemporaryDirectory(prefix="distributed-faiss-sift1m-") as temp_dir:
        storage_dir = Path(temp_dir) / "indexes"
        discovery_path = Path(temp_dir) / "servers.txt"
        discovery_path.write_text(
            str(args.num_servers) + "\n" + "".join(f"localhost,{port}\n" for port in ports)
        )

        try:
            for rank, port in enumerate(ports):
                server = IndexServer(rank, index_storage_dir=str(storage_dir))
                thread = threading.Thread(
                    target=server.start_blocking,
                    args=(port,),
                    daemon=True,
                )
                thread.start()
                servers.append(server)
                threads.append(thread)

            for server in servers:
                if not server.ready.wait(timeout=args.timeout_seconds):
                    raise TimeoutError("A shard server did not become ready")

            client = IndexClient(str(discovery_path))
            if args.omp_threads is not None:
                client.set_omp_num_threads(args.omp_threads)

            index_id = "sift1m_flat_l2"
            config = IndexCfg(
                index_builder_type="flat",
                dim=dimension,
                metric="l2",
                train_num=0,
                save_interval_sec=-1,
            )

            build_start = time.perf_counter()
            client.create_index(index_id, config)
            # Make data placement reproducible instead of using IndexClient's random start rank.
            client.cur_server_ids[index_id] = 0
            for start in range(0, base_count, args.add_batch_size):
                end = min(start + args.add_batch_size, base_count)
                vectors = np.ascontiguousarray(base[start:end], dtype=np.float32)
                ids = np.arange(start, end, dtype=np.int64).tolist()
                client.add_index_data(
                    index_id,
                    vectors,
                    ids,
                    train_async_if_triggered=False,
                )
            client.sync_train(index_id)
            wait_until_trained(client, index_id, args.timeout_seconds)
            build_seconds = time.perf_counter() - build_start

            shard_sizes = [server.get_ntotal(index_id) for server in servers]
            indexed_vectors = client.get_ntotal(index_id)

            result_ids = []
            search_start = time.perf_counter()
            for start in range(0, query_count, args.query_batch_size):
                end = min(start + args.query_batch_size, query_count)
                _, metadata = client.search(queries[start:end], args.k, index_id)
                result_ids.append(np.asarray(metadata, dtype=np.int64))
            search_seconds = time.perf_counter() - search_start
            actual_ids = np.concatenate(result_ids, axis=0)

            recall = recall_at_k(actual_ids, expected_ids, args.k)
            recall_at_1 = recall_at_k(actual_ids[:, :1], expected_ids, 1)
            valid_ids = bool(np.all((actual_ids >= 0) & (actual_ids < base_count)))
            passed = (
                indexed_vectors == base_count
                and sum(shard_sizes) == base_count
                and all(size >= args.k for size in shard_sizes)
                and valid_ids
                and recall >= 0.999
            )

            result = {
                "system": "meta-distributed-faiss",
                "dataset": "SIFT1M" if base_count == full_base_count else "SIFT1M-prefix",
                "dataset_dir": str(args.dataset_dir.resolve()),
                "index": "flat",
                "metric": "squared_l2",
                "reference_source": reference_source,
                "num_servers": args.num_servers,
                "num_vectors": base_count,
                "num_queries": query_count,
                "dimension": dimension,
                "k": args.k,
                "add_batch_size": args.add_batch_size,
                "query_batch_size": args.query_batch_size,
                "omp_threads_per_server": args.omp_threads,
                "shard_sizes": shard_sizes,
                "indexed_vectors": indexed_vectors,
                "recall_at_1": recall_at_1,
                "recall_at_k": recall,
                "valid_result_ids": valid_ids,
                "load_and_reference_seconds": load_seconds,
                "build_seconds": build_seconds,
                "search_seconds": search_seconds,
                "qps": query_count / search_seconds,
                "passed": passed,
            }
            rendered = json.dumps(result, indent=2)
            print(rendered)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered + "\n")

            if not passed:
                raise SystemExit("SIFT1M distributed Flat-L2 benchmark failed validation")
        finally:
            if client is not None:
                client.close()
            for server in servers:
                server.stop()
            for thread in threads:
                thread.join(timeout=1)


if __name__ == "__main__":
    main()
