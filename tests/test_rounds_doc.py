"""Drift gate for docs/reference/rounds.md.

The rounds page restates, in prose and a markdown table, values that live in
``whestbench.budget.ROUNDS``. Restating them is the point — a participant should
not have to import anything to look up what round they are in — but a restated
number is a number that can rot, and this one rots silently: nothing else in the
kit fails when whestbench advances a round.

So every cell of that table is asserted against the API here. When whestbench cuts
a new round, this test fails and names the row to update, rather than leaving the
kit advertising a rulebook that is no longer live.

Deliberately asserts the *rendered* strings rather than re-deriving them: the
failure mode being guarded against is the doc disagreeing with the code, and a test
that recomputes both sides from the same source would not catch it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from whestbench.budget import CURRENT_ROUND, ROUNDS

DOC = Path(__file__).resolve().parent.parent / "docs" / "reference" / "rounds.md"

#: Column order of the side-by-side table, left to right after the label column.
COLUMN_ORDER = ["v1-warmup", "v1-phase1", "v2-phase2"]


def _text() -> str:
    return DOC.read_text(encoding="utf-8")


def _normalize(cell: str) -> str:
    """Strip markdown emphasis and collapse whitespace, so `**1024 × 16**` == `1024 × 16`."""
    cell = cell.replace("**", "").replace("`", "")
    return re.sub(r"\s+", " ", cell).strip()


def _side_by_side_table() -> dict[str, list[str]]:
    """The 'Every round, side by side' table as {row label: [warmup, phase1, phase2]}."""
    body = _text().split("## Every round, side by side", 1)[1].split("\n###", 1)[0]
    rows: dict[str, list[str]] = {}
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [_normalize(c) for c in line.strip("|").split("|")]
        if len(cells) != 4 or set(cells[0]) <= {"-", ":", " "}:
            continue
        rows[cells[0]] = cells[1:]
    return rows


def _forward_pass_flops(width: int, depth: int) -> int:
    """One Monte-Carlo forward pass: a width x width matvec per layer, 2 FLOPs per MAC."""
    return depth * 2 * width * width


def test_doc_exists_and_covers_every_round():
    rows = _side_by_side_table()
    assert rows, "could not parse the side-by-side table out of rounds.md"
    assert set(COLUMN_ORDER) == set(ROUNDS), (
        f"rounds.md documents {COLUMN_ORDER} but whestbench.budget.ROUNDS has "
        f"{sorted(ROUNDS)} — a round was added or renamed upstream"
    )
    assert rows["Dataset tag"] == COLUMN_ORDER


def test_current_round_is_marked_current():
    rows = _side_by_side_table()
    header_line = next(
        line for line in _text().splitlines() if line.strip().startswith("| |") and "v1-warmup" in line
    )
    marked = [tag for tag in COLUMN_ORDER if f"{tag}` **(current)**" in header_line]
    assert marked == [CURRENT_ROUND.tag], (
        f"rounds.md marks {marked} as current; whestbench.budget.CURRENT_ROUND is "
        f"{CURRENT_ROUND.tag!r}"
    )
    assert f"you are being scored under `{CURRENT_ROUND.tag}`" in _text()
    assert rows["Dataset tag"][COLUMN_ORDER.index(CURRENT_ROUND.tag)] == CURRENT_ROUND.tag


@pytest.mark.parametrize("tag", COLUMN_ORDER)
def test_shape_and_predict_shape_match_the_api(tag):
    r = ROUNDS[tag]
    col = COLUMN_ORDER.index(tag)
    rows = _side_by_side_table()
    assert rows["Shape (width × depth)"][col] == f"{r.width} × {r.depth}"
    assert rows["predict() returns"][col] == f"({r.depth}, {r.width})"


@pytest.mark.parametrize("tag", COLUMN_ORDER)
def test_forward_pass_cost_and_budget_ratio(tag):
    r = ROUNDS[tag]
    col = COLUMN_ORDER.index(tag)
    rows = _side_by_side_table()

    fwd = _forward_pass_flops(r.width, r.depth)
    assert rows["One forward pass (2·d·w²)"][col] == f"{fwd:,} FLOPs"

    # The load-bearing claim of the page: every round buys ~65k forward passes.
    ratio = r.flop_budget / fwd
    documented = rows["Budget ÷ forward pass"][col].lstrip("≈ ").replace(",", "")
    assert int(documented) == round(ratio), (
        f"{tag}: doc says budget/forward-pass is {documented}, computed {ratio:,.0f}"
    )
    assert 64_000 <= ratio <= 66_000, (
        f"{tag} buys {ratio:,.0f} forward passes — the '~65,000 every round' claim in "
        f"rounds.md no longer holds and the section must be rewritten, not renumbered"
    )


@pytest.mark.parametrize("tag", COLUMN_ORDER)
def test_budget_lambda_and_caps_match_the_api(tag):
    r = ROUNDS[tag]
    col = COLUMN_ORDER.index(tag)
    rows = _side_by_side_table()

    budget_cell = rows["Budget B_m"][col].replace(",", "")
    assert str(r.flop_budget) in budget_cell or f"{r.flop_budget:.3g}".replace(
        "e+", "e"
    ) in budget_cell.replace("e+", "e"), (
        f"{tag}: budget cell {budget_cell!r} does not name {r.flop_budget}"
    )

    assert rows["Ground-truth samples N"][col] == f"{r.n_samples:.0e}".replace("e+09", "e9")

    lam = rows["λ (residual rate)"][col]
    if r.lambda_flops_per_second:
        assert f"{r.lambda_flops_per_second:g}".replace("e+", "e") in lam.replace("e+", "e")
    else:
        assert "0" in lam and "deprecated" in lam

    resid = rows["Residual cap"][col]
    if r.residual_wall_time_limit_s is None:
        assert resid == "none"
    else:
        assert resid == f"{r.residual_wall_time_limit_s} s"

    assert rows["Wall cap per predict()"][col] == f"{r.wall_time_limit_s:g} s"
    assert rows["Residual model"][col] == r.residual_mode


@pytest.mark.parametrize("tag", COLUMN_ORDER)
def test_effective_compute_column_matches_the_residual_model(tag):
    r = ROUNDS[tag]
    col = COLUMN_ORDER.index(tag)
    cell = _side_by_side_table()["Effective compute"][col]
    assert cell == ("C = F" if r.residual_mode == "gated" else "C = F + λR")


def test_current_round_block_matches_the_api():
    """The 'round you are being scored under' code block, which a reader trusts most."""
    block = _text().split("## The round you are being scored under", 1)[1].split("```", 2)[1]
    r = CURRENT_ROUND
    assert f"width {r.width} x depth {r.depth}" in block
    assert f"({r.depth}, {r.width})" in block
    assert f"{r.flop_budget:,}" in block
    assert f"{r.residual_wall_time_limit_s} s per MLP" in block
    assert f"{r.wall_time_limit_s:g} s" in block
    assert ("C_m = F_m" in block) == (r.residual_mode == "gated")


def test_ratio_table_agrees_with_the_side_by_side_table():
    """The '~65,000 forward passes' mini-table restates a row; keep the two in step."""
    body = _text().split("### The budget has always bought", 1)[1].split("\n\n", 3)[2]
    mini = {}
    for line in body.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [_normalize(c) for c in line.strip("|").split("|")]
        if len(cells) == 2 and cells[0] in ROUNDS:
            mini[cells[0]] = cells[1]
    assert set(mini) == set(ROUNDS), f"mini ratio table covers {sorted(mini)}"
    for tag, shown in mini.items():
        r = ROUNDS[tag]
        expected = round(r.flop_budget / _forward_pass_flops(r.width, r.depth))
        assert int(shown.replace(",", "")) == expected, f"{tag}: {shown} != {expected:,}"


def test_replay_recipe_restores_every_setting_of_the_round_it_names():
    """A partial replay recipe silently mixes two rulebooks — the page says so; enforce it."""
    recipe = _text().split("## Reproducing a score from an earlier round", 1)[1].split("```", 2)[1]
    tag = next(t for t in ROUNDS if f"@{t}" in recipe)
    r = ROUNDS[tag]
    assert f"--flop-budget {r.flop_budget}" in recipe
    lam = f"{r.lambda_flops_per_second:g}".replace("e+", "e")
    assert f"--lambda-flops-per-second {lam}" in recipe
    assert f"--wall-time-limit {r.wall_time_limit_s:g}" in recipe
    assert (r.residual_wall_time_limit_s is None) == ("--no-residual-wall-time-limit" in recipe)


def test_derived_rows_are_labelled_as_derived_not_read_from_the_api():
    """The page is the kit's tie-breaker for contradictory numbers, so its sourcing
    claim has to be exact. `RoundConfig` carries no forward-pass field — those two
    rows are computed here, and the page must not claim otherwise."""
    from dataclasses import fields

    assert not any(
        "forward" in f.name for f in fields(CURRENT_ROUND)
    ), "RoundConfig grew a forward-pass field — rounds.md can now source that row from the API"

    text = _text()
    assert "except the two forward-pass rows" in text, (
        "rounds.md claims every value is read from ROUNDS; two rows are derived"
    )
    assert "`2·d·w²`" in text, "the forward-pass row must be labelled as the arithmetic count"


def test_metered_forward_pass_figure_agrees_with_local_engine_api():
    """rounds.md quotes flopscope's *metered* cost so a reader who checks their own run
    is not left thinking the tie-breaker page is the outlier. That figure is restated
    from local-engine-api.md; pin the two together so they cannot drift apart."""
    rounds = _text()
    sibling = (DOC.parent / "local-engine-api.md").read_text(encoding="utf-8")

    metered = re.search(r"\*\*([\d,]+) FLOPs\*\* per pass at 1024×16", rounds)
    passes = re.search(r"giving \*\*([\d,]+)\*\*\s*\n?passes per budget", rounds)
    assert metered and passes, "rounds.md no longer states the metered per-pass figure"

    assert metered.group(1) in sibling, (
        f"rounds.md says {metered.group(1)} FLOPs/pass; local-engine-api.md does not agree"
    )
    assert passes.group(1) in sibling, (
        f"rounds.md says {passes.group(1)} passes; local-engine-api.md does not agree"
    )

    # And the pair must be self-consistent against the budget it divides.
    flops = int(metered.group(1).replace(",", ""))
    assert round(CURRENT_ROUND.flop_budget / flops) == int(passes.group(1).replace(",", ""))
