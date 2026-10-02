"""Raw target log probabilities with the same FP32 reduction tree at every batch size."""

import numpy as np
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _parts(X, P, STRIDE: tl.constexpr, V: tl.constexpr, T: tl.constexpr, B: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    cols = part * B + tl.arange(0, B)
    x = tl.load(X + row * STRIDE + cols, cols < V, other=-float("inf")).to(tl.float32)
    peak = tl.max(x, 0)
    mass = tl.sum(tl.exp(x - tl.where(peak == -float("inf"), 0.0, peak)), 0)
    tl.store(P + (row * T + part) * 2, peak)
    tl.store(P + (row * T + part) * 2 + 1, mass)


@tr.jit
def _finish(P, L, T: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    at = tl.arange(0, B)
    peak = tl.load(P + (row * T + at) * 2, at < T, other=-float("inf"))
    mass = tl.load(P + (row * T + at) * 2 + 1, at < T, other=0.0)
    maximum = tl.max(peak, 0)
    total = tl.sum(mass * tl.exp(peak - maximum), 0)
    tl.store(L + row, maximum + tl.log(total))


ENTRIES = 128 * 1024**2 // 48   # vocabulary entries a call holds: 128 MiB of temporaries at up to 48 bytes an entry


def log_sum_exp(logits: torch.Tensor) -> torch.Tensor:
    """Each row's log-sum-exp in FP32 [n], by the same reduction tree whatever the row count."""

    n, vocab = logits.shape
    tiles = tr.cdiv(vocab, 1024)
    parts = torch.empty((n, tiles, 2), dtype=torch.float32, device=logits.device)
    lse = torch.empty((n,), dtype=torch.float32, device=logits.device)
    _parts[(n, tiles)](logits, parts, logits.stride(0), vocab, tiles, 1024, num_warps=4)
    _finish[(n,)](parts, lse, tiles, tr.next_power_of_2(tiles), num_warps=4)
    return lse


def top_columns(logits: torch.Tensor, count: int) -> torch.Tensor:
    """Each row's ``count`` largest columns, highest first and ties to the lower column, at any row count."""

    values = logits.float()
    bits = values.view(torch.int32).to(torch.int64)
    bits = torch.where(values == 0, 0, bits)
    ordered = torch.where(bits < 0, ~bits, bits ^ 0x80000000) - 0x80000000
    token_ids = torch.arange(logits.shape[1], dtype=torch.int64, device=logits.device)
    keys = (ordered << 32) | (0xFFFFFFFF - token_ids)
    return keys.topk(count, dim=-1, sorted=True).indices


@torch.no_grad()
def prompt_rows(logits: torch.Tensor, targets: list[int], top: int, gather) -> list[tuple[float, int, list]]:
    """One rank's vocabulary shard of some rows [n, shard] -> each row's (target log probability, its rank in the whole
    vocabulary, the top ``top`` (id, log probability) highest first), the same on every rank.

    The shards are equal and in rank order: gather slot r holds ids r * shard onward. Every rank sends its logit at
    column ``target % shard`` and the combiner reads slot ``target // shard``'s, so a second gather can count every
    shard's logits at or above it: vLLM's rank of a prompt token (its top entries are ranked by position)."""

    n, shard = logits.shape
    if not logits.is_cuda or logits.stride(1) != 1 or len(targets) != n:
        raise ValueError("prompt rows need CUDA logits [rows, shard] and one target a row")
    dev = logits.device
    count = min(top, shard)
    ids = torch.tensor(targets, dtype=torch.int64, device=dev)
    rows = torch.arange(n, device=dev)
    cols = top_columns(logits, count) if count else torch.empty((n, 0), dtype=torch.int64, device=dev)
    packed = torch.cat([log_sum_exp(logits)[:, None], logits.gather(1, cols).float(),
                        cols.to(torch.int32).view(torch.float32), logits[rows, ids % shard].float()[:, None]], dim=1)
    got = gather(packed.contiguous().view(-1)).view(-1, n, 2 + 2 * count)           # [world, n, width]
    world = got.shape[0]
    if min(targets, default=0) < 0 or max(targets, default=0) >= world * shard:
        raise ValueError("a prompt token is outside the vocabulary")
    target = got[ids // shard, rows, -1]
    above = (logits.float() >= target[:, None]).sum(dim=1).to(torch.int32)
    ranks = gather(above.view(torch.float32)).view(world, n).view(torch.int32).sum(dim=0).tolist()
    g = got.cpu()
    parts = g[:, :, 0].double().numpy()                                            # [world, n]
    peak = parts.max(axis=0)
    lse = peak + np.log(np.exp(parts - peak).sum(axis=0))
    values = np.concatenate(list(g[:, :, 1:1 + count].double().numpy()), axis=1)   # [n, world * count]
    columns = g[:, :, 1 + count:1 + 2 * count].contiguous().view(torch.int32).numpy().astype(np.int64)
    gids = np.concatenate([columns[r] + r * shard for r in range(world)], axis=1)
    chosen = target.double().cpu().numpy() - lse
    out = []
    for i in range(n):
        order = np.lexsort((gids[i], -values[i]))[:count]
        out.append((float(chosen[i]), int(ranks[i]), [(int(gids[i, j]), float(values[i, j] - lse[i])) for j in order]))
    return out


@torch.no_grad()
def capture(logits, tokens, positions, probabilities, rows=None):
    """Only accepted target rows reach the collector; source logits are read-only."""

    if probabilities is None or not tokens:
        return
    # Bound all vocabulary-sized sorting temporaries, including selected source rows.
    width = max(1, ENTRIES // logits.shape[1])
    if len(tokens) > width:
        for start in range(0, len(tokens), width):
            end = start + width
            capture(logits[start:end] if rows is None else logits, tokens[start:end], positions[start:end],
                    probabilities, None if rows is None else rows[start:end])
        return
    if rows is not None:
        logits = logits.index_select(0, torch.tensor(rows, dtype=torch.long, device=logits.device))
    if not logits.is_cuda or logits.ndim != 2 or logits.shape[0] != len(tokens) or logits.stride(1) != 1:
        raise ValueError("probabilities need CUDA target rows and one accepted token per row")
    lse = log_sum_exp(logits)
    ids = torch.tensor(tokens, dtype=torch.long, device=logits.device)[:, None]
    chosen = (logits.gather(1, ids).float()[:, 0] - lse).cpu().tolist()
    labels = getattr(probabilities, "labels", None)
    if labels:                                   # a decision: the label logits at the prompt's last position
        picked = logits.index_select(1, torch.tensor(labels, dtype=torch.long, device=logits.device)).float()
        for row, pos in enumerate(positions):
            probabilities.add_labels(pos, picked[row].cpu().tolist(), float(lse[row].item()))
    count = min(probabilities.top, logits.shape[1])
    if count:
        top_ids = top_columns(logits, count)
        scores = (logits.gather(1, top_ids).float() - lse[:, None]).cpu().tolist()
        alternatives = top_ids.cpu().tolist()
    else:
        alternatives, scores = [[] for _ in tokens], [[] for _ in tokens]
    probabilities.add(positions, tokens, chosen, alternatives, scores)
