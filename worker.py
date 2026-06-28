"""worker.py — RQ worker process for libaix background jobs."""

from __future__ import annotations

import os

from rq import Connection, Worker

from job_queue import _QUEUE_NAME, get_redis


def main() -> None:
    conn = get_redis()
    with Connection(conn):
        worker = Worker([_QUEUE_NAME])
        worker.work(with_scheduler=True)


if __name__ == "__main__":
    os.environ.setdefault("LIBAIX_QUEUE", "libaix")
    main()
