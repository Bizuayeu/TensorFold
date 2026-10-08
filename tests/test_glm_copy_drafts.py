"""Copy drafts (prompt lookup, MiaAI-Lab patches 0007 and 0032's first half): what a context proposes. Pure numpy."""

from __future__ import annotations

import pytest

from tensorfold.families.glm5_next.cuda import copy_drafts as cd

PROMPT = list(range(100, 140))                     # 40 distinct tokens


def drafts(context, prompt=None):
    return cd.CopyDrafts(context, len(context) - 1 if prompt is None else prompt)


def test_the_constants():
    from tensorfold.families.glm5_next.cuda.engine import GRAPH_ROWS, MAX_ROWS

    assert (cd.MATCH, cd.REPLY_MATCH, cd.MOST) == (8, 16, 5)
    assert 1 <= cd.MISS_MOST <= cd.MOST
    # a copied round's window (the pending token and its drafts) replays a captured graph
    assert cd.MOST + 1 <= max(GRAPH_ROWS) <= MAX_ROWS


def test_no_proposal_when_the_suffix_never_occurred():
    assert drafts(PROMPT + [7]).propose() == []
    assert drafts(PROMPT[:5] + [7]).propose() == []          # shorter than the match
    assert drafts([]).propose() == []


def test_a_prompt_occurrence_of_eight_tokens_proposes_what_followed():
    # the prompt's tokens 10..16 and then the pending token: the context's last 8 occurred at 10
    context = PROMPT + PROMPT[10:17] + [PROMPT[17]]
    assert drafts(context).propose() == PROMPT[18:23]          # MOST = 5
    assert drafts(context).propose(room=2) == PROMPT[18:20]
    assert drafts(context).propose(room=0) == []


def test_seven_matching_tokens_are_not_enough():
    context = PROMPT + [1] + PROMPT[11:17] + [PROMPT[17]]      # one token off within the last 8
    assert drafts(context).propose() == []


def test_the_latest_occurrence_with_room_after_it_wins():
    tail = list(range(500, 508))
    a, b = [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]
    context = tail + a + tail + b + [99] + tail
    assert drafts(context, prompt=len(context)).propose() == b
    # the latest occurrence (13) has 1 token after its match before the context ends, the one at 0 has 14
    nines = [9] * 8
    context = nines + a + nines + [9]
    assert drafts(context, prompt=len(context)).propose() == a
    # none has 5 after it: what follows the earliest, to the context's end
    assert drafts([9] * 10, prompt=10).propose() == [9, 9]


def test_an_occurrence_inside_the_reply_needs_sixteen_tokens():
    loop = list(range(300, 312))
    prompt = PROMPT
    # the reply repeats a 12-token loop: its last 8 occur inside the reply after one lap, 16 only after two
    one_lap = drafts(prompt + loop + loop[:8], prompt=len(prompt))
    assert one_lap.propose() == []
    two_laps = drafts(prompt + loop + loop + loop[:4], prompt=len(prompt))
    assert two_laps.propose() == loop[4:9]
    # the same 8 tokens in the prompt are enough there
    from_prompt = drafts(prompt + loop + loop[:8], prompt=len(prompt) + len(loop))
    assert from_prompt.propose() == loop[8:12] + loop[:1]


def test_a_reply_occurrence_without_sixteen_tokens_before_its_end_is_skipped():
    loop = list(range(300, 312))
    # the whole context is reply: the occurrence at 0 has no 8 tokens before it, the one at 12 has
    assert drafts(loop + loop[:8], prompt=0).propose() == []
    assert drafts(loop + loop + loop[:8], prompt=0).propose() == loop[8:12] + loop[:1]
    # the 8 tokens before a reply occurrence may be prompt; they must match all the same
    assert drafts(PROMPT[:8] + loop + loop[:8], prompt=8).propose() == []


def test_after_a_missed_copy_round_miss_most_drafts_until_one_keeps_all(monkeypatch):
    """MISS_MOST is MOST, so a miss cuts nothing; set below it, the copies after a miss propose that many."""

    monkeypatch.setattr(cd, "MISS_MOST", 3)
    tail = list(range(500, 508))
    run = list(range(600, 640))
    c = drafts(tail + run + tail, prompt=len(tail) + len(run) + len(tail))
    assert c.propose() == run[:5]
    c.extend(run[:5])                    # kept 4 of the 5 drafts and the sample after them: a miss
    assert c.missed
    c.extend(tail)                       # an MTP round between (no proposal): the miss stands
    assert c.propose() == run[:3]
    c.extend(run[:4])                    # every draft kept, and the sample after them
    assert not c.missed
    c.extend(tail)
    assert len(c.propose()) == cd.MOST


def test_a_round_without_a_proposal_settles_nothing():
    tail = list(range(500, 508))
    run = list(range(600, 640))
    c = drafts(tail + run + tail, prompt=len(tail) + len(run) + len(tail))
    assert c.propose() == run[:5]
    c.extend(run[:6])                    # all kept
    assert not c.missed
    c.extend([1])                        # an MTP round's tokens: no copy proposal was made
    assert not c.missed


def test_extend_grows_the_buffer():
    c = drafts([1, 2, 3])
    c.extend(list(range(5000)))
    assert c.length == 5003 and c.buf[:c.length].tolist() == [1, 2, 3] + list(range(5000))


def test_the_switch_is_on_unless_zero():
    assert cd.enabled({}) and cd.enabled({"TF_GLM_COPY_DRAFTS": ""}) and cd.enabled({"TF_GLM_COPY_DRAFTS": " 1 "})
    assert not cd.enabled({"TF_GLM_COPY_DRAFTS": "0"})


@pytest.mark.parametrize("value", ["2", "on", "true", "off"])
def test_other_values_are_refused(value):
    with pytest.raises(ValueError, match="TF_GLM_COPY_DRAFTS is 0 or 1"):
        cd.enabled({"TF_GLM_COPY_DRAFTS": value})
