"""GLM's idle doorbell (engine._ring / _await_bell) on a real localhost TCPStore, the ranks as threads (CPU)."""

from __future__ import annotations

import socket
import threading
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch.distributed import TCPStore  # noqa: E402

from tensorfold.families.glm5_next.cuda.engine import GlmEngine  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _bind(r):
    for name in ("_store", "_bell_key", "_ring", "_await_bell"):
        setattr(r, name, getattr(GlmEngine, name).__get__(r))
    return r


def _ranks(world=2):
    port = _free_port()
    master = TCPStore("127.0.0.1", port, world, True, timeout=timedelta(seconds=30), wait_for_workers=False)
    stores = [master] + [TCPStore("127.0.0.1", port, world, False, timeout=timedelta(seconds=30))
                         for _ in range(world - 1)]
    return (master, *(_bind(SimpleNamespace(rank=r, world=world, comm=SimpleNamespace(store=s)))
                      for r, s in enumerate(stores)))


def test_rank1_blocks_until_rank0_rings_then_follows_every_request_in_order():
    master, r0, r1 = _ranks()
    woke: list[tuple[int, float]] = []

    def follower():
        for _ in range(3):
            r1._await_bell()
            woke.append((r1._bell, time.monotonic()))

    t = threading.Thread(target=follower)
    t.start()
    time.sleep(0.3)
    assert woke == []                                   # nothing rung: rank 1 is still waiting
    rang = []
    for _ in range(3):
        rang.append(time.monotonic())
        r0._ring()
        time.sleep(0.1)
    t.join(10)
    assert not t.is_alive()
    assert [n for n, _ in woke] == [1, 2, 3]
    assert all(w >= r for (_, w), r in zip(woke, rang))
    assert r0._bell == 3
    assert master.num_keys() <= 2                       # consumed keys are deleted (the store's own key(s) remain)


def test_each_of_two_followers_takes_every_bell():
    """Three ranks: rank 0 rings each follower's own key, so neither deletes the bell the other still waits for."""

    master, r0, r1, r2 = _ranks(3)
    woke: dict[int, list[int]] = {1: [], 2: []}

    def follower(r):
        for _ in range(3):
            r._await_bell()
            woke[r.rank].append(r._bell)

    threads = [threading.Thread(target=follower, args=(r,), daemon=True) for r in (r1, r2)]
    for t in threads:
        t.start()
    for _ in range(3):
        r0._ring()
        time.sleep(0.1)
    for t in threads:
        t.join(10)
    assert not any(t.is_alive() for t in threads), woke
    assert woke == {1: [1, 2, 3], 2: [1, 2, 3]}
    assert [r0._bell_key(r, 1) for r in (1, 2)] == ["tf_glm_request_1_1", "tf_glm_request_2_1"]
    assert master.num_keys() <= 3                       # every consumed key is deleted


def test_two_ranks_keep_their_bell_key():
    _, r0, _ = _ranks(2)
    assert r0._bell_key(1, 7) == "tf_glm_request_7"     # a rank 1 of an older build waits on this name


def test_rings_before_rank1_waits_are_not_lost():
    _, r0, r1 = _ranks()
    r0._ring()
    r0._ring()                                          # rank 1 still busy with an earlier request
    r1._await_bell()
    r1._await_bell()
    assert r1._bell == 2


def test_no_store_means_no_doorbell():
    r = _bind(SimpleNamespace(rank=1, world=2, comm=SimpleNamespace()))
    r._ring()
    r._await_bell()                                     # returns at once
    assert not hasattr(r, "_bell")


def test_idle_timeouts_are_retried_and_other_errors_raise():
    class Store:
        def __init__(self, errors):
            self.errors, self.deleted = list(errors), []

        def wait(self, keys, timeout):
            if self.errors:
                raise self.errors.pop(0)

        def delete_key(self, key):
            self.deleted.append(key)

    store = Store([RuntimeError("Socket Timeout"), RuntimeError("wait timeout after 3600000ms")])
    r = _bind(SimpleNamespace(rank=1, world=2, comm=SimpleNamespace(store=store)))
    r._await_bell()                                     # two idle hours, then the request
    assert r._bell == 1 and store.deleted == ["tf_glm_request_1"]
    r.comm.store = Store([RuntimeError("Connection reset by peer")])
    with pytest.raises(RuntimeError, match="Connection reset"):
        r._await_bell()                                 # rank 0 is gone: rank 1 stops instead of waiting forever
