"""A GLM tool call the model's end token left open (written before ``</tool_call>``) is closed and sent as a call when it
then parses whole; otherwise its markup stays the reply's text. A call whose ``<arg_key>`` the model wrote as
whitespace gets it back in both servers' parsers. A call the token limit cut, and every other reply, is unchanged.
The replies through the CUDA server's GlmApp are in test_glm_tool_call_end_app.py."""

from __future__ import annotations

import json

import pytest

from tensorfold.cuda.reply_text import close_glm_call, parse_tool_calls
from tensorfold.server.tools import parse_tool_calls_from_content, repair_glm_keys

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "limit": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "run", "parameters": {"type": "object", "properties": {
        "command": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "now", "parameters": {"type": "object", "properties": {}}}},
]
PARSERS = pytest.mark.parametrize("parser", [parse_tool_calls, parse_tool_calls_from_content],
                                  ids=["cuda-server", "http-server"])


def args(text, tools=TOOLS, parser=parse_tool_calls, **kw):
    return [(c["function"]["name"], json.loads(c["function"]["arguments"]))
            for c in parser(text, tools, **kw)[1] or []]


# -- the missing <arg_key> --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("block, repaired", [
    ("read_file path</arg_key><arg_value>/x</arg_value>",
     "read_file <arg_key>path</arg_key><arg_value>/x</arg_value>"),
    ("read_file\npath</arg_key><arg_value>/x</arg_value>",
     "read_file\n<arg_key>path</arg_key><arg_value>/x</arg_value>"),
    ("read_file<arg_key>path</arg_key><arg_value>/x</arg_value>\nlimit</arg_key><arg_value>3</arg_value>",
     "read_file<arg_key>path</arg_key><arg_value>/x</arg_value>\n<arg_key>limit</arg_key><arg_value>3</arg_value>"),
    ("read_file path</arg_key><arg_value>/x</arg_value> limit</arg_key><arg_value>3</arg_value>",
     "read_file <arg_key>path</arg_key><arg_value>/x</arg_value> <arg_key>limit</arg_key><arg_value>3</arg_value>"),
])
def test_a_missing_arg_key_is_put_back(block, repaired):
    assert repair_glm_keys(block) == repaired
    assert repair_glm_keys(repaired) == repaired


@pytest.mark.parametrize("block", [
    "read_file<arg_key>path</arg_key><arg_value>/x</arg_value>",      # nothing missing
    "read_file",
    "read_filepath</arg_key><arg_value>/x</arg_value>",               # no space: no name left before the key
    " path</arg_key><arg_value>/x</arg_value>",                       # no name at all
    '{"name": "run", "arguments": {"command": "ls"}}',
])
def test_blocks_without_a_missing_key_are_unchanged(block):
    assert repair_glm_keys(block) == block


@PARSERS
def test_both_parsers_read_a_repaired_call(parser):
    # before: the first came back as text, the second lost its second argument (or was refused as one call)
    text = "<tool_call>read_file filePath</arg_key><arg_value>/home/a/main.go</arg_value></tool_call>"
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {
        "filePath": {"type": "string"}}}}}]
    assert args(text, tools, parser) == [("read_file", {"filePath": "/home/a/main.go"})]
    assert args(text, tools, parser, max_calls=1) == [("read_file", {"filePath": "/home/a/main.go"})]
    later = ("<tool_call>read_file<arg_key>path</arg_key><arg_value>a</arg_value>\nlimit</arg_key><arg_value>3"
             "</arg_value></tool_call>")
    assert args(later, parser=parser) == [("read_file", {"path": "a", "limit": 3})]
    assert args(later, parser=parser, max_calls=1) == [("read_file", {"path": "a", "limit": 3})]


# -- closing a call the end token left open ------------------------------------------------------------------------

@pytest.mark.parametrize("text, suffix", [
    ("<tool_call>read_file<arg_key>path</arg_key><arg_value>x</arg_value>", "</tool_call>"),
    ("<tool_call>run<arg_key>command</arg_key><arg_value>ls -la", "</arg_value></tool_call>"),
    ("Let me look:<tool_call>now", "</tool_call>"),
    ("<tool_call>now</tool_call>\n<tool_call>read_file<arg_key>path</arg_key><arg_value>x</arg_value>\n",
     "</tool_call>"),
    ("<tool_call>read_file path</arg_key><arg_value>x", "</arg_value></tool_call>"),       # and repaired
])
def test_a_call_that_parses_once_closed_is_closed(text, suffix):
    assert close_glm_call(text, TOOLS) == suffix
    assert args(text + suffix)[-1][0] in ("read_file", "run", "now")


@pytest.mark.parametrize("text", [
    "no call at all",
    "<tool_call>now</tool_call>",                                       # closed already
    "<tool_call>launch<arg_key>x</arg_key><arg_value>1",                # a tool the client did not offer
    "<tool_call>run<arg_key>comm",                                      # a key with no value
    "<tool_call>run<arg_key>command</arg_key>",                         # ... or no value yet
    "<tool_call>run command=ls",                                        # not GLM's markup
    "<tool_call>",                                                      # nothing written
    "<tool_call>now</think>The answer.",                                # the think block went on past it
    "<tool_call>run<arg_key>command</arg_key><arg_value>ls</arg_value> and then",       # text after an argument
])
def test_a_call_that_would_not_parse_is_left_open(text):
    assert close_glm_call(text, TOOLS) == ""
