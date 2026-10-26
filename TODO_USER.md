# What you still have to do

The code, figures, validation, and `report/report.pdf` are in this folder.
The GitHub title-block fields are filled in as `Sbboss` / `mlds490-hw1` /
`https://github.com/Sbboss/mlds490-hw1`.

1. Read the AI-assistance paragraph in the report and confirm it matches what
   you actually did. Edit it if it does not. The draft says the implementation
   and the runs were produced with Cursor, and that you are responsible for the
   numbers and for being able to explain the code. `STUDY_GUIDE.md` is the
   explanation.

2. Submit on Canvas, before Mon 2026-10-26 11:59 pm CT:
   - the PDF (`report/report.pdf`)
   - the GitHub link `https://github.com/Sbboss/mlds490-hw1`

## Rebuild the PDF after you edit the report

```bash
cd "/Users/sbboss/NU/Q4/MLDS-490/HW#1"
.venv/bin/python scripts/build_report.py
```

Open `report/report.pdf` and check that every figure is visible and no table
runs off the page.

## What is in the GitHub repo

Code (`shuttle/`, `pg/`, `validation/`, `scripts/`), `requirements.txt`,
`README.md`, `DECISIONS.md`, `STUDY_GUIDE.md`, figures, the report, and the
CSV/JSON result files the report quotes. Large Pong `final.pt` weight files
and intermediate checkpoints are not pushed (they stay local / on deepdish).
CartPole `final.pt` files are tiny and are included.
