"""PC: python -I -B test_gverdict.py [unittest's arguments] -- gverdict.py's lines on fake grids: a counted cell gets
fsum.py's word, and a cell some arm did not count (round_ms None, as TP=2's 1016K x4) gets no verdict instead of
stopping the cells after it.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gverdict  # noqa: E402


def write(bench, label, cells):
    with open(os.path.join(bench, "%s-grid.json" % label), "w", encoding="utf-8") as f:
        json.dump({"cells": [{"context": c, "streams": n, "round_ms": v} for (c, n), v in cells.items()]}, f)


def lines(bench, mains, heads):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        gverdict.main([bench, mains, heads])
    return out.getvalue().splitlines()


class Cells(unittest.TestCase):
    def test_counted_and_uncounted_cells(self):
        with tempfile.TemporaryDirectory() as bench:
            write(bench, "ma", {(262144, 1): 42.403, (1040384, 1): 46.704, (1040384, 4): None})
            write(bench, "hd", {(262144, 1): 42.063, (1040384, 1): 44.878, (1040384, 4): 97.862})
            write(bench, "mb", {(262144, 1): 42.208, (1040384, 1): 46.293, (1040384, 4): 97.000})
            got = lines(bench, "ma,mb", "hd")
        self.assertEqual(got, [
            "256K x1  main 42.403 42.208  head 42.063  -0.57%  main's range 0.46%  faster",
            "1016K x1  main 46.704 46.293  head 44.878  -3.49%  main's range 0.88%  faster",
            "1016K x4  main - 97.000  head 97.862  no verdict: a cell did not count",
        ])

    def test_head_cell_uncounted(self):
        with tempfile.TemporaryDirectory() as bench:
            write(bench, "ma", {(262144, 4): 80.0})
            write(bench, "hd", {(262144, 4): None})
            write(bench, "mb", {(262144, 4): 80.2})
            got = lines(bench, "ma,mb", "hd")
        self.assertEqual(got, ["256K x4  main 80.000 80.200  head -  no verdict: a cell did not count"])


if __name__ == "__main__":
    unittest.main()
