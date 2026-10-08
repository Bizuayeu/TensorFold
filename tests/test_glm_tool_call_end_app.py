"""Through the CUDA server's GlmApp, streamed or not: a tool call the end token left open is sent as a call when it
parses whole (inside an unclosed think block too) and stays text when it would not; one the token limit or a stop
string cut is not closed; a closed call missing its ``<arg_key>`` is a call. These need tokenizers and jinja2; the
parsers' tests (test_glm_tool_call_end.py) do not, so they are skipped apart."""

from __future__ import annotations

import json
import threading

import pytest

tokenizers = pytest.importorskip("tokenizers")
pytest.importorskip("jinja2")

from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from tensorfold.cuda import server
from tensorfold.families.glm5_next.cuda.app import GlmApp, ThinkingOffTemplate
from tests.test_cuda_admission import http_server, post
from tests.test_cuda_stop_strings import GlmEngine, events, write_template
from tests.test_glm_tool_call_end import TOOLS

END = "<|observation|>"
MARKS = ("<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "<think>",
         "</think>")


def glm_tokenizer() -> Tokenizer:
    """One token a byte, plus GLM's call and think marks and an end token as single tokens (as GLM-5.3's are)."""

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tok = Tokenizer(models.BPE(vocab={ch: i for i, ch in enumerate(alphabet)}, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens([END, *MARKS])
    return tok


TOK = glm_tokenizer()
END_ID = TOK.token_to_id(END)


class Engine(GlmEngine):
    eos = (END_ID,)


def make_app(tmp_path, reply_ids, thinking):
    write_template(tmp_path)
    app = GlmApp.__new__(GlmApp)
    app.engine = Engine(reply_ids)
    app.served = "fake-glm"
    app.tok = TOK
    app.template = ThinkingOffTemplate(server.ChatTemplate(tmp_path))
    app.default_thinking = thinking
    app.sampling = {"temperature": 0.0}
    app.max_tokens = 4096
    app.native_context_window = app.context_window = 0
    app.lock = threading.Lock()
    return app


def _reply(tmp_path, written, streamed, *, ended=True, thinking=False, **fields):
    """(content, reasoning, calls, finish) of a reply that writes ``written`` (then the end token when ``ended``)."""

    reply_ids = TOK.encode(written, add_special_tokens=False).ids + ([END_ID] if ended else [])
    app = make_app(tmp_path, reply_ids, thinking)
    body = {"messages": [{"role": "user", "content": "x"}], "tools": TOOLS, "temperature": 0, "stream": streamed,
            **fields}
    with http_server(app) as port:
        status, text = post(port, body, True)
    assert status == 200, text
    if not streamed:
        choice = json.loads(text)["choices"][0]
        message = choice["message"]
        calls = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in message.get("tool_calls") or []]
        return message.get("content") or "", message.get("reasoning_content") or "", calls, choice["finish_reason"]
    content, reasoning, calls, finish = "", "", {}, None
    for chunk in events(text):
        if not chunk.get("choices"):
            continue
        delta = chunk["choices"][0].get("delta", {})
        content += delta.get("content") or ""
        reasoning += delta.get("reasoning_content") or ""
        for t in delta.get("tool_calls", []):
            entry = calls.setdefault(t["index"], ["", ""])
            entry[0] += t["function"].get("name") or ""
            entry[1] += t["function"].get("arguments") or ""
        finish = chunk["choices"][0].get("finish_reason") or finish
    return content, reasoning, [(n, json.loads(a)) for n, a in (calls[i] for i in sorted(calls))], finish


OPEN = [   # (the reply, ended by the end token; the calls; the content)
    ("Reading it.<tool_call>read_file<arg_key>path</arg_key><arg_value>notes.txt</arg_value>",
     [("read_file", {"path": "notes.txt"})], "Reading it."),
    ("<tool_call>run<arg_key>command</arg_key><arg_value>ls -la", [("run", {"command": "ls -la"})], ""),
    ("The time:<tool_call>now", [("now", {})], "The time:"),
    ("<tool_call>now</tool_call><tool_call>run<arg_key>command</arg_key><arg_value>date</arg_value>",
     [("now", {}), ("run", {"command": "date"})], ""),
    ("<tool_call>read_file path</arg_key><arg_value>notes.txt</arg_value>", [("read_file", {"path": "notes.txt"})], ""),
]


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("written, calls, content", OPEN, ids=range(len(OPEN)))
def test_a_call_the_end_token_left_open_is_a_call(tmp_path, written, calls, content, streamed):
    got_content, _, got_calls, finish = _reply(tmp_path, written, streamed)
    assert (got_content.strip(), got_calls, finish) == (content, calls, "tool_calls")


@pytest.mark.parametrize("streamed", [False, True])
def test_a_call_at_the_end_of_an_unclosed_think_block_is_a_call(tmp_path, streamed):
    got = _reply(tmp_path, "I need the file.<tool_call>read_file<arg_key>path</arg_key><arg_value>x</arg_value>",
                 streamed, thinking=True)
    assert got == ("", "I need the file.", [("read_file", {"path": "x"})], "tool_calls")


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("written", [
    "<tool_call>launch_rocket<arg_key>when</arg_key><arg_value>now</arg_value>",   # a tool not offered
    "<tool_call>run<arg_key>comm",                                                 # a key with no value
    "<tool_call>",
])
def test_a_call_left_open_that_would_not_parse_stays_text(tmp_path, written, streamed):
    content, _, calls, finish = _reply(tmp_path, "Trying." + written, streamed)
    assert (content, calls, finish) == ("Trying." + written, [], "stop")


@pytest.mark.parametrize("streamed", [False, True])
def test_a_call_the_token_limit_cut_is_not_closed(tmp_path, streamed):
    written = "Removing it. <tool_call>run<arg_key>command</arg_key><arg_value>rm -rf build"
    content, _, calls, finish = _reply(tmp_path, written, streamed, ended=False,
                                       max_tokens=len(TOK.encode(written, add_special_tokens=False).ids))
    assert (content, calls, finish) == (written, [], "length")


@pytest.mark.parametrize("streamed", [False, True])
def test_a_call_a_stop_string_cut_is_not_closed(tmp_path, streamed):
    written = "Removing it. <tool_call>run<arg_key>command</arg_key><arg_value>rm -rf build STOP and more"
    content, _, calls, finish = _reply(tmp_path, written, streamed, stop=" STOP")
    assert (content, calls, finish) == (written[:written.find(" STOP")], [], "stop")


@pytest.mark.parametrize("streamed", [False, True])
def test_a_closed_call_with_a_missing_key_is_a_call(tmp_path, streamed):
    got = _reply(tmp_path, "<tool_call>read_file path</arg_key><arg_value>notes.txt</arg_value></tool_call>", streamed)
    assert got == ("", "", [("read_file", {"path": "notes.txt"})], "tool_calls")
