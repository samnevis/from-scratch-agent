from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from katas.bank import SYNTH_JSONL, freeze_eval_ids
from katas.schema import Kata
from sandbox.runner import run_tests

TEMPLATES = [
    "add",
    "mul",
    "clamp_const",
    "starts",
    "count_char",
    "sum_range",
    "is_divisible",
    "repeat_str",
    "list_head",
    "abs_diff",
]


def _verified(kata: Kata) -> Kata:
    vis = run_tests(kata.solution, kata.tests)
    hid = run_tests(kata.solution, kata.hidden_tests)
    if not vis.ok or not hid.ok:
        raise RuntimeError(f"unverified {kata.id}: vis={vis.summary()} hid={hid.summary()}")
    return kata


def make_one(rng: random.Random, i: int, split: str) -> Kata:
    kind = TEMPLATES[i % len(TEMPLATES)]
    kid = f"synth_{i:05d}"
    if kind == "add":
        a, b = rng.randint(-20, 20), rng.randint(-20, 20)
        c, d = rng.randint(-20, 20), rng.randint(-20, 20)
        return _verified(
            Kata(
                id=kid,
                prompt="Write add(x, y) returning x + y.",
                entry_point="add",
                tests=[f"assert add({a}, {b}) == {a + b}"],
                hidden_tests=[f"assert add({c}, {d}) == {c + d}"],
                solution="def add(x, y):\n    return x + y\n",
                split=split,
                tags=["math", "synth"],
            )
        )
    if kind == "mul":
        a, b = rng.randint(-10, 10), rng.randint(-10, 10)
        return _verified(
            Kata(
                id=kid,
                prompt="Write mul(x, y) returning x * y.",
                entry_point="mul",
                tests=[f"assert mul({a}, {b}) == {a * b}"],
                hidden_tests=[f"assert mul({a + 1}, {b - 1}) == {(a + 1) * (b - 1)}"],
                solution="def mul(x, y):\n    return x * y\n",
                split=split,
                tags=["math", "synth"],
            )
        )
    if kind == "clamp_const":
        lo, hi = 0, rng.randint(5, 20)
        x = rng.randint(-5, hi + 5)
        y = rng.randint(-5, hi + 5)
        return _verified(
            Kata(
                id=kid,
                prompt=f"Write clamp(x) clipping x into [{lo}, {hi}].",
                entry_point="clamp",
                tests=[f"assert clamp({x}) == {max(lo, min(hi, x))}"],
                hidden_tests=[f"assert clamp({y}) == {max(lo, min(hi, y))}"],
                solution=f"def clamp(x):\n    return max({lo}, min({hi}, x))\n",
                split=split,
                tags=["math", "synth"],
            )
        )
    if kind == "starts":
        prefix = rng.choice(["ab", "he", "py", "go"])
        ok = prefix + rng.choice(["x", "y", "zz"])
        bad = rng.choice(["zz", "nope", "q"])
        return _verified(
            Kata(
                id=kid,
                prompt=f"Write starts(s) True iff s starts with {prefix!r}.",
                entry_point="starts",
                tests=[f"assert starts({ok!r}) is True", f"assert starts({bad!r}) is False"],
                hidden_tests=[f"assert starts({prefix!r}) is True"],
                solution=f"def starts(s):\n    return s.startswith({prefix!r})\n",
                split=split,
                tags=["strings", "synth"],
            )
        )
    if kind == "count_char":
        ch = rng.choice(list("abcxyz"))
        s1 = "".join(rng.choice("abcxyz") for _ in range(8))
        s2 = "".join(rng.choice("abcxyz") for _ in range(8))
        return _verified(
            Kata(
                id=kid,
                prompt=f"Write count_ch(s) counting occurrences of {ch!r}.",
                entry_point="count_ch",
                tests=[f"assert count_ch({s1!r}) == {s1.count(ch)}"],
                hidden_tests=[f"assert count_ch({s2!r}) == {s2.count(ch)}"],
                solution=f"def count_ch(s):\n    return s.count({ch!r})\n",
                split=split,
                tags=["strings", "synth"],
            )
        )
    if kind == "sum_range":
        n = rng.randint(3, 12)
        m = rng.randint(3, 12)
        return _verified(
            Kata(
                id=kid,
                prompt="Write sum_to(n) returning 1+...+n for n>=1.",
                entry_point="sum_to",
                tests=[f"assert sum_to({n}) == {n * (n + 1) // 2}"],
                hidden_tests=[f"assert sum_to({m}) == {m * (m + 1) // 2}"],
                solution="def sum_to(n):\n    return n * (n + 1) // 2\n",
                split=split,
                tags=["math", "synth"],
            )
        )
    if kind == "is_divisible":
        k = rng.choice([2, 3, 4, 5])
        a, b = rng.randint(0, 40), rng.randint(0, 40)
        return _verified(
            Kata(
                id=kid,
                prompt=f"Write is_div(n) True iff n is divisible by {k}.",
                entry_point="is_div",
                tests=[f"assert is_div({a}) is {a % k == 0}"],
                hidden_tests=[f"assert is_div({b}) is {b % k == 0}"],
                solution=f"def is_div(n):\n    return n % {k} == 0\n",
                split=split,
                tags=["math", "synth"],
            )
        )
    if kind == "repeat_str":
        s = rng.choice(["ab", "x", "ha"])
        n, m = rng.randint(1, 4), rng.randint(1, 4)
        return _verified(
            Kata(
                id=kid,
                prompt=f"Write repeat(n) returning {s!r} repeated n times.",
                entry_point="repeat",
                tests=[f"assert repeat({n}) == {s * n!r}"],
                hidden_tests=[f"assert repeat({m}) == {s * m!r}"],
                solution=f"def repeat(n):\n    return {s!r} * n\n",
                split=split,
                tags=["strings", "synth"],
            )
        )
    if kind == "list_head":
        xs = [rng.randint(0, 9) for _ in range(rng.randint(2, 5))]
        ys = [rng.randint(0, 9) for _ in range(rng.randint(2, 5))]
        return _verified(
            Kata(
                id=kid,
                prompt="Write head(xs) returning the first element of a non-empty list.",
                entry_point="head",
                tests=[f"assert head({xs}) == {xs[0]}"],
                hidden_tests=[f"assert head({ys}) == {ys[0]}"],
                solution="def head(xs):\n    return xs[0]\n",
                split=split,
                tags=["lists", "synth"],
            )
        )
    # abs_diff
    a, b = rng.randint(-20, 20), rng.randint(-20, 20)
    c, d = rng.randint(-20, 20), rng.randint(-20, 20)
    return _verified(
        Kata(
            id=kid,
            prompt="Write abs_diff(a, b) returning the absolute difference |a-b|.",
            entry_point="abs_diff",
            tests=[f"assert abs_diff({a}, {b}) == {abs(a - b)}"],
            hidden_tests=[f"assert abs_diff({c}, {d}) == {abs(c - d)}"],
            solution="def abs_diff(a, b):\n    return abs(a - b)\n",
            split=split,
            tags=["math", "synth"],
        )
    )


def generate(n_train: int = 200, n_eval: int = 100, seed: int = 0) -> list[Kata]:
    rng = random.Random(seed)
    items: list[Kata] = []
    for i in range(n_train):
        items.append(make_one(rng, i, "train"))
    eval_items = []
    for i in range(n_train, n_train + n_eval):
        eval_items.append(make_one(rng, i, "agent_eval"))
    items.extend(eval_items)
    return items


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=int, default=200)
    p.add_argument("--eval", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=SYNTH_JSONL)
    args = p.parse_args()
    items = generate(args.train, args.eval, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for k in items:
            f.write(json.dumps(k.to_dict()) + "\n")
    eval_ids = [k.id for k in items if k.split == "agent_eval"]
    freeze_eval_ids(eval_ids)
    train_ids = {k.id for k in items if k.split == "train"}
    overlap = train_ids & set(eval_ids)
    if overlap:
        raise SystemExit(f"train/eval leak: {overlap}")
    print(f"wrote {args.out} n={len(items)} eval={len(eval_ids)}")


if __name__ == "__main__":
    main()
