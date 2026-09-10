"""Running the same numpy work on several cores.

Threads, not processes. The work a search spends its time on — inverse FFTs,
matrix products, reductions over long arrays — happens inside numpy, which
drops the GIL while it runs, so threads scale on it about as well as processes
do. What they do not do is copy the data: a drum's recordings are over a
hundred megabytes, and handing that to eight worker processes (which is what
Windows does, having no fork) costs more than the parallelism returns.
"""

from __future__ import annotations

import contextlib
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def _single_threaded_blas():
    """Keep numpy's own threads out of the way while ours are running.

    numpy ships a threaded BLAS, so a matrix product inside a worker thread
    tries to use every core on its own — and eight workers each doing that on
    a sixteen-core machine is a hundred and twenty-eight threads fighting over
    sixteen cores. `threadpoolctl` is the way to say no; without it installed
    the searches still work, a little slower, and OPENBLAS_NUM_THREADS=1 in the
    environment does the same job.
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return contextlib.nullcontext()
    return threadpool_limits(limits=1, user_api="blas")


def resolve_jobs(jobs: int) -> int:
    """How many workers `jobs` asks for. 0 or less means one per core."""
    return max(1, jobs if jobs > 0 else (os.cpu_count() or 1))


def map_workers(
    function: Callable[[T], R], items: Sequence[T] | Iterable[T], jobs: int = 1
) -> list[R]:
    """`function` over `items`, in order, on `jobs` threads."""
    items = list(items)
    workers = min(resolve_jobs(jobs), len(items))
    if workers <= 1 or len(items) <= 1:
        return [function(item) for item in items]

    with _single_threaded_blas(), ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, items))
