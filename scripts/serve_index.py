#!/usr/bin/env python3

"""Run one distributed-faiss index server in its own process."""

import argparse
import signal
from pathlib import Path

from distributed_faiss.server import IndexServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--storage-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.rank < 0 or not 1 <= args.port <= 65535:
        raise ValueError("--rank must be nonnegative and --port must be between 1 and 65535")

    args.storage_dir.mkdir(parents=True, exist_ok=True)
    server = IndexServer(args.rank, index_storage_dir=str(args.storage_dir))
    signal.signal(signal.SIGTERM, lambda *_: server.stop())
    try:
        server.start_blocking(args.port)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
