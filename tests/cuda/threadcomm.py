"""``comm.NCCL`` for ranks that are threads of one process on one GPU, any rank count (tests only).

Each all-gather hands its input over in host memory: the rank finishes its queued work, every rank's input is
copied into the output in rank order, and no rank goes on before all have copied. The exchanges cannot be captured
in CUDA graphs (they wait on the host). The type follows ``tests/cuda/threadcomm.py`` of ashhart/TensorFold PR #159
(drowzeys, Apache-2.0), written for this tree with the all-gather only.
"""

from __future__ import annotations

import threading

import torch

TIMEOUT = 300           # s a rank waits for the others at one exchange (test_flashnext_tp's _Hub)


class Store:
    """The rendezvous store's keys for the engine's idle doorbell, in memory. ``close`` ends every wait for a key not
    set (a follower then leaves ``follow``); ``open`` lets them wait again."""

    def __init__(self) -> None:
        self.keys: set[str] = set()
        self.closed = False
        self.cond = threading.Condition()

    def set(self, key: str, value) -> None:
        with self.cond:
            self.keys.add(key)
            self.cond.notify_all()

    def wait(self, keys: list[str], timeout) -> None:
        with self.cond:
            self.cond.wait_for(lambda: self.closed or all(k in self.keys for k in keys), timeout.total_seconds())
            if all(k in self.keys for k in keys):
                return
            raise Closed("the store is closed") if self.closed else RuntimeError("wait timeout")

    def delete_key(self, key: str) -> None:
        with self.cond:
            self.keys.discard(key)

    def open(self) -> None:
        with self.cond:
            self.closed = False

    def close(self) -> None:
        with self.cond:
            self.closed = True
            self.cond.notify_all()


class Closed(Exception):
    pass


class Hub:
    def __init__(self, world: int) -> None:
        self.world = world
        self.slots: list = [None] * world
        self.barrier = threading.Barrier(world, timeout=TIMEOUT)


class ThreadComm:
    def __init__(self, hub: Hub, rank: int, store=None) -> None:
        self.hub, self.rank, self.world = hub, rank, hub.world
        if store is not None:
            self.store = store

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        if recv.numel() != send.numel() * self.world or send.dtype != recv.dtype:
            raise ValueError("all_gather: recv must hold world x send of the same dtype")
        torch.cuda.current_stream().synchronize()
        self.hub.slots[self.rank] = send
        self.hub.barrier.wait()
        n = send.numel()
        flat = recv.view(-1)
        for r in range(self.world):
            flat[r * n:(r + 1) * n].copy_(self.hub.slots[r].reshape(-1))
        torch.cuda.current_stream().synchronize()
        self.hub.barrier.wait()

    def barrier(self) -> None:
        self.hub.barrier.wait()

    def ready(self, label: str, **kwargs) -> None:
        self.hub.barrier.wait()


def run_ranks(fn, world: int, hub: Hub | None = None, store: Store | None = None) -> list:
    """fn(rank, comm) on every rank at once (threads); results in rank order. A rank's error breaks the others' waits
    and is raised here."""

    hub = hub or Hub(world)
    results: list = [None] * world
    errors: list = []

    def body(r: int) -> None:
        try:
            with torch.no_grad():
                results[r] = fn(r, ThreadComm(hub, r, store))
        except BaseException as exc:        # noqa: BLE001  (raised below; unblock the other ranks)
            errors.append(exc)
            hub.barrier.abort()

    threads = [threading.Thread(target=body, args=(r,)) for r in range(world)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if errors:              # the rank that failed first, not the others' broken waits
        raise next((e for e in errors if not isinstance(e, threading.BrokenBarrierError)), errors[0])
    return results
