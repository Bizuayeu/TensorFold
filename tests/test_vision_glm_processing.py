"""GLM image prompt expansion follows its processor grid and keeps the image tower CPU-prepared."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tensorfold.vision.glm_processing import GLMImageProcessor


class Tokenizer:
    def convert_tokens_to_ids(self, token):
        return {"<|image|>": 10}.get(token)

    def __call__(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False, "return_attention_mask": False}
        out = []
        while text:
            marker = next((x for x in ("<|begin_of_image|>", "<|image|>", "<|end_of_image|>")
                           if text.startswith(x)), None)
            if marker:
                out.append({"<|begin_of_image|>": 8, "<|image|>": 10, "<|end_of_image|>": 9}[marker])
                text = text[len(marker):]
            else:
                out.append(ord(text[0]))
                text = text[1:]
        return {"input_ids": out}


class Processor:
    image_token = "<|image|>"

    def __init__(self, grids):
        self.tokenizer = Tokenizer()
        self.image_processor = ImageBatchProcessor(grids)
        self.grids = grids
        self.calls = []

    def replace_image_token(self, image_inputs, image_idx):
        count = int(np.prod(image_inputs["image_grid_thw"][image_idx])) // self.image_processor.merge_size**2
        return self.image_token * count


class ImageBatchProcessor:
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2

    def __init__(self, grids):
        self.grids, self.calls = grids, []

    def __call__(self, images, return_tensors=None, max_image_tokens=None, min_image_tokens=None):
        idx = len(self.calls)
        self.calls.append((images, return_tensors, max_image_tokens, min_image_tokens))
        grid = self.grids[idx]
        count = int(np.prod(grid))
        return {"pixel_values": np.zeros((count, 3 * 2 * 14 * 14), dtype=np.float32),
                "image_grid_thw": np.asarray([grid], dtype=np.int64)}


def image(content_hash="img", detail="auto"):
    return SimpleNamespace(content_hash=content_hash, detail=detail, to_pil=lambda: content_hash)


CONFIG = {"model_type": "glm5_next", "image_token_id": 10,
          "vision_config": {"out_hidden_size": 6, "patch_size": 14, "temporal_patch_size": 2,
                            "spatial_merge_size": 2, "hidden_size": 4, "intermediate_size": 8, "depth": 2}}


@pytest.mark.parametrize(("setting", "value"), [("patch_size", 7), ("temporal_patch_size", 1), ("merge_size", 4)])
def test_glm_image_processor_rejects_geometry_that_disagrees_with_the_tower(setting, value):
    processor = Processor([[1, 4, 4]])
    setattr(processor.image_processor, setting, value)
    with pytest.raises(ValueError, match="disagrees with the vision tower"):
        GLMImageProcessor(CONFIG, processor)


@pytest.mark.parametrize("grid", [[2, 4, 4], [1, 3, 4], [1, -4, -4]])
def test_glm_image_prompt_rejects_non_image_or_unaligned_grids(grid):
    front = GLMImageProcessor(CONFIG, Processor([grid]))
    with pytest.raises(ValueError, match="one frame and merge-aligned positive dimensions"):
        front.prepare("<|begin_of_image|><|image|><|end_of_image|>", [image()])


@pytest.mark.parametrize("budget", [1, 4, 15])
def test_glm_image_prompt_respects_budgets_smaller_than_sixteen(budget):
    processing = pytest.importorskip("mlx_vlm.models.glm5_next.processing")
    pil = pytest.importorskip("PIL.Image")
    processor = SimpleNamespace(tokenizer=Tokenizer(), image_token="<|image|>",
                                image_processor=processing.Glm5NextImageProcessor())
    front = GLMImageProcessor(CONFIG, processor)
    source = SimpleNamespace(content_hash="small-image", detail="auto",
                             to_pil=lambda: pil.new("RGB", (28, 28)))
    prepared = front.prepare("<|begin_of_image|><|image|><|end_of_image|>", [source],
                             max_visual_tokens=budget)
    assert 1 <= prepared.visual_tokens <= budget


def test_glm_image_prompt_expands_patch_grid_and_limits_total_visual_tokens():
    processor = Processor([[1, 4, 4], [1, 2, 4]])
    front = GLMImageProcessor({"model_type": "glm5_next", "image_token_id": 10,
                              "vision_config": {"out_hidden_size": 6, "patch_size": 14,
                                                "temporal_patch_size": 2, "spatial_merge_size": 2,
                                                "hidden_size": 4, "intermediate_size": 8, "depth": 2}}, processor)
    prepared = front.prepare("question<|begin_of_image|><|image|><|end_of_image|> and "
                             "<|begin_of_image|><|image|><|end_of_image|>",
                             [image("a"), image("b", "low")], max_visual_tokens=32, max_prompt_tokens=32)
    assert prepared.token_ids.count(10) == 6
    assert prepared.visual_tokens == 6
    assert prepared.image_hashes == ("a", "b")
    assert prepared.image_grid_thw.tolist() == [[1, 4, 4], [1, 2, 4]]
    assert prepared.pixel_values.shape == (24, 3 * 2 * 14 * 14)
    assert all(not a.flags.writeable for a in (prepared.pixel_values, prepared.image_grid_thw))
    assert [call[2] for call in processor.image_processor.calls] == [16, 16]


def test_glm_image_prompt_refuses_marker_count_and_context_overflow():
    processor = Processor([[1, 4, 4]])
    front = GLMImageProcessor({"model_type": "glm5_next", "image_token_id": 10,
                                  "vision_config": {"out_hidden_size": 6, "patch_size": 14,
                                                    "temporal_patch_size": 2, "spatial_merge_size": 2}}, processor)
    with pytest.raises(ValueError, match="one image marker"):
        front.prepare("no image here", [image()])
    with pytest.raises(ValueError, match="maximum context length is 5 tokens: the expanded image prompt"):
        front.prepare("<|begin_of_image|><|image|><|end_of_image|>", [image()], max_prompt_tokens=5)


SPAN = "<|begin_of_image|><|image|><|end_of_image|>"      # what GLM's chat template writes for a picture


def test_a_prompt_without_a_quoted_marker_expands_as_it_always_has():
    # the token ids of a picture request that quotes no marker are pinned: the quoted-marker escape leaves them be
    front = GLMImageProcessor(CONFIG, Processor([[1, 4, 4]]))
    assert front.prepare(f"Q{SPAN}A", [image()]).token_ids == (81, 8, 10, 10, 10, 10, 9, 65)
    front = GLMImageProcessor(CONFIG, Processor([[1, 4, 4], [1, 2, 4]]))
    text = f"question{SPAN} and <|begin_of_image|> said {SPAN}"
    parts = text.split("<|image|>")                       # the expansion before quoted markers were told apart
    expanded = parts[0] + "".join("<|image|>" * n + rest for n, rest in zip((4, 2), parts[1:]))
    assert front.prepare(text, [image("a"), image("b")]).token_ids == tuple(Tokenizer()(
        expanded, add_special_tokens=False, return_attention_mask=False)["input_ids"])


def test_a_marker_the_conversation_quotes_stays_text_beside_a_real_picture():
    front = GLMImageProcessor(CONFIG, Processor([[1, 4, 4]]))
    prepared = front.prepare(f"the log said <|image|> twice: <|image|>{SPAN}?", [image()])
    assert prepared.token_ids.count(10) == prepared.visual_tokens == 4
    assert prepared.image_spans == ((len("the log said <|\u200bimage|> twice: <|\u200bimage|>") + 1,
                                     len("the log said <|\u200bimage|> twice: <|\u200bimage|>") + 5),)
    with pytest.raises(ValueError, match="one image marker"):     # a quoted whole span cannot be told from a picture
        front.prepare(f"quoted {SPAN} then {SPAN}", [image()])
    with pytest.raises(ValueError, match="one image marker"):     # nor can a picture the template left out
        front.prepare("only a quote <|image|>", [image()])


def test_a_history_quoting_a_marker_with_a_picture_is_not_refused():
    from tensorfold.server.prompts import prepare_images

    pil = pytest.importorskip("PIL.Image")
    import base64
    import io

    png = io.BytesIO()
    pil.new("RGB", (2, 2), "red").save(png, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(png.getvalue()).decode()
    messages = [{"role": "user", "content": "what does <|image|> in this log mean?"},
                {"role": "assistant", "content": "It is GLM's picture token."},
                {"role": "user", "content": [{"type": "text", "text": "and this one?"},
                                             {"type": "image_url", "image_url": {"url": url}}]}]

    def render(template):                                  # each message's text, a picture's span in its place
        return "".join(m["content"] if isinstance(m["content"], str) else
                       "".join(p["text"] if p["type"] == "text" else SPAN for p in m["content"]) for m in template)

    rendered = prepare_images(GLMImageProcessor(CONFIG, Processor([[1, 4, 4]])), messages, render)
    assert rendered.tokens.count(10) == rendered.vision.visual_tokens == 4


def test_a_request_without_pictures_never_reaches_the_frontend():
    # its prompt is the template's text tokenized as it is, a quoted marker included (both servers branch on this)
    from tensorfold.server.prompts import has_images, prepare_prompt

    messages = [{"role": "user", "content": "what does <|image|> mean?"},
                {"role": "user", "content": [{"type": "text", "text": "<|begin_of_image|><|image|>"}]}]
    assert not has_images(messages)
    app = SimpleNamespace(render=lambda messages, tools, thinking: ([1, 10, 2], 3), vision=None)
    assert prepare_prompt(app, messages, None, False, None, {}).tokens == [1, 10, 2]
