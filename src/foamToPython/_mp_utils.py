from multiprocessing import get_context
from typing import Optional


def get_spawn_pool(processes: Optional[int] = None):
    ctx = get_context("spawn")
    return ctx.Pool(processes=processes)
