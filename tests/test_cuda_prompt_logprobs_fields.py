"""CUDA completions take a prompt as token ids and answer vLLM's ``prompt_logprobs``; unsupported or malformed
requests fail before generation."""

import json

import pytest

from tests.test_cuda_admission import http_server
from tests.test_cuda_server_errors import Engine, app_for, request

IDS = [1, 2, 1]


def prompt_app(tmp_path, supported=True):
    from tokenizers import Tokenizer, decoders, models

    class Target(Engine):
        supports_prompt_logprobs = supported

        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, prompt_logprobs=None):
            self.calls.append({"prompt": list(prompt), "rows": prompt_logprobs})
            if prompt_logprobs is not None:
                prompt_logprobs.add(1, -0.5, 1, [(2, -0.5), (1, -1.25)])
                prompt_logprobs.add(2, -3.0, 4, [(2, -0.25), (0, -1.0)])
            on_tokens([1])
            return {}

    app = app_for(tmp_path)
    app.engine = Target()
    app.tok = Tokenizer(models.WordLevel({"[UNK]": 0, "A": 1, "ĠB": 2}, unk_token="[UNK]"))
    app.tok.decoder = decoders.ByteLevel()
    return app


def test_token_id_prompts_reach_the_engine_as_sent(tmp_path):
    app = prompt_app(tmp_path)
    with http_server(app) as port:
        status, _, text = request(port, {"prompt": IDS, "max_tokens": 1, "temperature": 0}, chat=False)
    assert status == 200, text
    assert app.engine.calls == [{"prompt": IDS, "rows": None}]
    assert "prompt_logprobs" not in json.loads(text)["choices"][0]


def test_prompt_logprobs_read_as_vllm_completions(tmp_path):
    app = prompt_app(tmp_path)
    with http_server(app) as port:
        status, _, text = request(port, {"prompt": IDS, "max_tokens": 1, "temperature": 0, "seed": 42,
                                         "prompt_logprobs": 2}, chat=False)
    assert status == 200, text
    assert app.engine.calls[0]["rows"].top == 2
    choice = json.loads(text)["choices"][0]
    assert choice["text"] == "A"
    assert choice["prompt_logprobs"] == [
        None,
        {"2": {"logprob": -0.5, "rank": 1, "decoded_token": " B"},
         "1": {"logprob": -1.25, "rank": 2, "decoded_token": "A"}},
        {"1": {"logprob": -3.0, "rank": 4, "decoded_token": "A"},
         "2": {"logprob": -0.25, "rank": 1, "decoded_token": " B"},
         "0": {"logprob": -1.0, "rank": 2, "decoded_token": "[UNK]"}}]


@pytest.mark.parametrize("fields, chat, name", [
    ({"prompt": [1, 3]}, False, "0 to 2"), ({"prompt": [1, -1]}, False, "0 to 2"),
    ({"prompt": [1, True]}, False, "integer token ids"), ({"prompt": [1, 1.0]}, False, "integer token ids"),
    ({"prompt": []}, False, "empty"), ({"prompt": 5}, False, "token ids"),
    ({"prompt": IDS, "prompt_logprobs": 21}, False, "prompt_logprobs"),
    ({"prompt": IDS, "prompt_logprobs": -1}, False, "prompt_logprobs"),
    ({"prompt": IDS, "prompt_logprobs": True}, False, "prompt_logprobs"),
    ({"prompt": IDS, "prompt_logprobs": 5, "stream": True}, False, "prompt_logprobs"),
    ({"messages": [{"role": "user", "content": "Hi"}], "prompt_logprobs": 5}, True, "prompt_logprobs")])
def test_malformed_requests_are_refused_before_generation(tmp_path, fields, chat, name):
    app = prompt_app(tmp_path)
    with http_server(app) as port:
        status, _, text = request(port, {"max_tokens": 1, **fields}, chat=chat)
    assert status == 400, text
    assert name in json.loads(text)["error"]["message"]
    assert app.engine.calls == []


def test_engines_without_prompt_rows_refuse_them(tmp_path):
    app = prompt_app(tmp_path, supported=False)
    with http_server(app) as port:
        status, _, text = request(port, {"prompt": IDS, "max_tokens": 1, "prompt_logprobs": 5}, chat=False)
    assert status == 400 and "not supported" in json.loads(text)["error"]["message"]
    assert app.engine.calls == []
