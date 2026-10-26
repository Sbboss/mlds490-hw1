"""Run the PG validation suite; write validation_pg_results.md/.json.

    python -m validation.run_validation            # fast + default
    python -m validation.run_validation --pong     # include Pong (needs ale-py)
    python -m validation.run_validation --all      # everything except T-B5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from validation.records import RECORDS, clear, snapshot
from validation.test_baseline import run_baseline_tests
from validation.test_gradient_exact import run_gradient_tests
from validation.test_returns import run_return_tests


def write_report(records, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validation_pg_results.json").write_text(json.dumps(records, indent=1))
    lines = [
        "| # | check | kind | result | measured | expected |",
        "|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(records, 1):
        if r["kind"] == "exact":
            res = "PASS" if r["passed"] else "FAIL"
        else:
            res = "PASS" if r["passed"] else "NOT HELD"
        # keep table cells one-line
        meas = r["measured"].replace("|", "/").replace("\n", " ")
        exp = r["expected"].replace("|", "/").replace("\n", " ")
        if len(meas) > 180:
            meas = meas[:177] + "..."
        lines.append(f"| {i} | {r['name']} | {r['kind']} | {res} | {meas} | {exp} |")
    text = "\n".join(lines) + "\n"
    (out_dir / "validation_pg_results.md").write_text(text)
    # also copy next to the shuttle file so the report can include both
    Path("validation_pg_results.md").write_text(text)
    Path("validation_pg_results.json").write_text(json.dumps(records, indent=1))
    return text


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pong", action="store_true", help="run T-P* (needs ale-py)")
    p.add_argument("--cartpole", action="store_true", default=True)
    p.add_argument("--no-cartpole", action="store_true")
    p.add_argument("--pong-skip-slow", action="store_true")
    p.add_argument("--all", action="store_true")
    args = p.parse_args(argv)
    if args.all:
        args.pong = True

    clear()
    print("=== T-R* estimator / returns ===", flush=True)
    run_return_tests()
    run_gradient_tests()
    print("=== T-B* baseline (fast) ===", flush=True)
    run_baseline_tests()
    if not args.no_cartpole:
        print("=== T-C* CartPole (fast) ===", flush=True)
        from validation.test_cartpole import run_cartpole_fast_tests
        run_cartpole_fast_tests()
    if args.pong:
        print("=== T-P* Pong ===", flush=True)
        from validation.test_pong import run_pong_tests
        run_pong_tests(skip_slow=args.pong_skip_slow)

    recs = snapshot()
    table = write_report(recs, Path("results/validation"))
    print(table)
    failed = [r for r in recs if r["kind"] == "exact" and not r["passed"]]
    held = [r for r in recs if r["kind"] == "prediction" and not r["passed"]]
    print(f"\n{len(recs)} checks: {len(failed)} exact failures, {len(held)} predictions not held")
    for r in failed:
        print(f"  FAIL {r['name']}: {r['measured']}", file=sys.stderr)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
