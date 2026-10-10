"""The CUDA server tokenizes through ``App._encode_ids``: HF tokenizers' batch call (the GIL released), the same ids
as ``encode`` for either ``add_special_tokens``; a stand-in without the batch call encodes as before."""

from __future__ import annotations

import ast
import inspect
import threading
import time

import pytest

tokenizers = pytest.importorskip("tokenizers")
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from tensorfold.cuda import server

WORDS = ["[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "hello", "world", "!", "\n"]


def tokenizer() -> Tokenizer:
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1),
                                                                                                ("[SEP]", 2)])
    return tok


def app(tok) -> server.App:
    a = object.__new__(server.App)
    a.tok = tok
    return a


@pytest.mark.parametrize("special", [False, True])
@pytest.mark.parametrize("text", ["", "hello world !", "a b c " * 500, "hello 世界 ́ zz", "\n\n"])
def test_the_same_ids_as_encode(text, special):
    tok = tokenizer()
    assert app(tok)._encode_ids(text, special) == tok.encode(text, add_special_tokens=special).ids


def test_the_flag_passes_through():
    a = app(tokenizer())
    assert a._encode_ids("a b", True) == [1, 3, 4, 2] and a._encode_ids("a b", False) == [3, 4]


def test_a_stand_in_without_the_batch_call_encodes():
    class StandIn:
        def __init__(self):
            self.calls = []

        def encode(self, text, add_special_tokens):
            self.calls.append((text, add_special_tokens))
            return type("E", (), {"ids": [len(text)]})()

    tok = StandIn()
    assert app(tok)._encode_ids("abc", True) == [3] and tok.calls == [("abc", True)]


def test_no_other_server_path_calls_encode():
    tree = ast.parse(inspect.getsource(server))
    helper = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_encode_ids")
    inside = {id(n) for n in ast.walk(helper)}
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "encode" and isinstance(n.func.value, ast.Attribute) and n.func.value.attr == "tok"
             and id(n) not in inside]
    assert calls == []


def counted_rate(fn) -> float:
    """A counting thread's increments a second while ``fn`` runs."""

    counts = {"n": 0}
    done = threading.Event()

    def count():
        while not done.is_set():
            counts["n"] += 1

    t = threading.Thread(target=count)
    t.start()
    try:
        time.sleep(0.05)
        start = counts["n"]
        began = time.perf_counter()
        fn()
        spent = time.perf_counter() - began
        ran = counts["n"] - start
    finally:
        done.set()
        t.join()
    return ran / spent


def test_a_long_encode_leaves_other_threads_running():
    """The batch call releases the GIL: a counting thread runs far more while a long text tokenizes than under
    ``encode``, which holds it."""

    tok = tokenizer()
    text = "hello world ! " * 400_000                                  # 1.2M words
    held = counted_rate(lambda: tok.encode(text, add_special_tokens=False))
    freed = counted_rate(lambda: app(tok)._encode_ids(text, False))
    assert freed > 5 * held
