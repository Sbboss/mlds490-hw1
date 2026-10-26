"""Shared record format: name, kind, passed, measured, expected, why."""
from __future__ import annotations

RECORDS: list[dict] = []


def record(name, kind, passed, measured, expected, why):
    RECORDS.append(dict(
        name=name, kind=kind, passed=bool(passed),
        measured=str(measured), expected=str(expected), why=why,
    ))


def clear():
    RECORDS.clear()


def snapshot():
    return list(RECORDS)
