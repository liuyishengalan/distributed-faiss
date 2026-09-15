#!/usr/bin/env python3

"""End-to-end correctness check for a sharded Flat-L2 index."""

import argparse
import json
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-servers", type=int, default=2)
    parser.add_argument("--num-vectors", type=int, default=10_000)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


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


def recall_at_k(actual, expected):
    hits = 0
    for actual_row, expected_row in zip(actual, expected):
        hits += len(set(actual_row).intersection(expected_row))
    return hits / expected.size


def wait_until_trained(client, index_id, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if client.get_state(index_id) == IndexState.TRAINED:
            return
        time.sleep(0.01)
    raise TimeoutError(f"Index {index_id!r} did not become trained")


def main():
    args = parse_args()
    if args.num_servers < 1:
        raise ValueError("--num-servers must be positive")
    if args.num_vectors < args.num_servers:
        raise ValueError("--num-vectors must be at least --num-servers")
    if not 1 <= args.k <= args.num_vectors:
        raise ValueError("--k must be between 1 and --num-vectors")

    rng = np.random.default_rng(args.seed)
    database = rng.standard_normal((args.num_vectors, args.dimension), dtype=np.float32)
    queries = rng.standard_normal((args.num_queries, args.dimension), dtype=np.float32)

    reference = faiss.IndexFlatL2(args.dimension)
    reference.add(database)
    reference_distances, reference_ids = reference.search(queries, args.k)

    ports = reserve_free_ports(args.num_servers)
    servers = []
    threads = []
    client = None

    with tempfile.TemporaryDirectory(prefix="distributed-faiss-smoke-") as temp_dir:
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
            index_id = "smoke_flat_l2"
            config = IndexCfg(
                index_builder_type="flat",
                dim=args.dimension,
                metric="l2",
                train_num=0,
                save_interval_sec=-1,
            )

            build_start = time.perf_counter()
            client.create_index(index_id, config)
            vector_chunks = np.array_split(database, args.num_servers)
            id_chunks = np.array_split(
                np.arange(args.num_vectors, dtype=np.int64), args.num_servers
            )
            for vector_chunk, id_chunk in zip(vector_chunks, id_chunks):
                client.add_index_data(
                    index_id,
                    vector_chunk,
                    id_chunk.tolist(),
                    train_async_if_triggered=False,
                )
            client.sync_train(index_id)
            wait_until_trained(client, index_id, args.timeout_seconds)
            build_seconds = time.perf_counter() - build_start

            shard_sizes = [server.get_ntotal(index_id) for server in servers]
            total_vectors = client.get_ntotal(index_id)

            search_start = time.perf_counter()
            distributed_distances, distributed_metadata = client.search(queries, args.k, index_id)
            search_seconds = time.perf_counter() - search_start
            distributed_ids = np.asarray(distributed_metadata, dtype=np.int64)

            recall = recall_at_k(distributed_ids, reference_ids)
            ids_match = np.array_equal(distributed_ids, reference_ids)
            distances_match = np.allclose(
                distributed_distances,
                reference_distances,
                rtol=1e-5,
                atol=1e-5,
            )
            passed = (
                total_vectors == args.num_vectors
                and all(size > 0 for size in shard_sizes)
                and ids_match
                and distances_match
                and recall == 1.0
            )

            result = {
                "system": "meta-distributed-faiss",
                "index": "flat",
                "metric": "squared_l2",
                "num_servers": args.num_servers,
                "num_vectors": args.num_vectors,
                "num_queries": args.num_queries,
                "dimension": args.dimension,
                "k": args.k,
                "shard_sizes": shard_sizes,
                "indexed_vectors": total_vectors,
                "recall_at_k": recall,
                "ids_match": ids_match,
                "distances_match": distances_match,
                "max_distance_error": float(
                    np.max(np.abs(distributed_distances - reference_distances))
                ),
                "build_seconds": build_seconds,
                "search_seconds": search_seconds,
                "qps": args.num_queries / search_seconds,
                "passed": passed,
            }
            print(json.dumps(result, indent=2))

            if not passed:
                raise SystemExit("Flat-L2 distributed smoke test failed")
        finally:
            if client is not None:
                client.close()
            for server in servers:
                server.stop()
            for thread in threads:
                thread.join(timeout=1)


if __name__ == "__main__":
    main()
