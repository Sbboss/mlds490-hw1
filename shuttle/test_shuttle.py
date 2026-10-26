"""pytest wrapper: every 'exact' validation check must pass."""
import pytest

from validation import run_all

RECORDS = run_all()


@pytest.mark.parametrize("rec", [r for r in RECORDS if r["kind"] == "exact"], ids=lambda r: r["name"])
def test_exact(rec):
    assert rec["passed"], f"{rec['name']}: measured {rec['measured']} expected {rec['expected']}"
