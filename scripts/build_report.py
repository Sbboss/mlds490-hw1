"""Build report/report.html and report/report.pdf.

The Anaconda pandoc in /opt/anaconda3/bin is an x86_64 binary and does not
run on this machine. This script is the rebuild path: Python-Markdown, images
inlined, then headless Chrome.
"""
from __future__ import annotations

import base64
import mimetypes
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "report" / "report.md"
HTML = ROOT / "report" / "report.html"
PDF = ROOT / "report" / "report.pdf"
CSS = (ROOT / "report" / "print.css").read_text()
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def _img(src: str) -> str:
    path = (ROOT / "report" / src).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def main() -> None:
    body = markdown.markdown(
        MD.read_text(),
        extensions=["tables", "fenced_code", "sane_lists", "toc"],
    )
    # markdown leaves image src as written in the file (../figures/...)
    def repl(m: re.Match) -> str:
        return f'src="{_img(m.group(1))}"'
    body = re.sub(r'src="([^"]+)"', repl, body)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>MLDS 490 Homework 1</title>
<style>{CSS}</style></head><body>{body}</body></html>
"""
    HTML.write_text(html)
    print("wrote", HTML, "bytes", HTML.stat().st_size)
    # Chrome chokes on a file URI that contains `#` (this homework folder name).
    # Print from a temp path with no special characters, then copy back.
    with tempfile.TemporaryDirectory(prefix="mlds490_report_") as tmp:
        tmp_html = Path(tmp) / "report.html"
        tmp_pdf = Path(tmp) / "report.pdf"
        tmp_html.write_text(html)
        subprocess.run(
            [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
             f"--print-to-pdf={tmp_pdf}", tmp_html.as_uri()],
            check=True,
        )
        shutil.copy2(tmp_pdf, PDF)
    print("wrote", PDF, "bytes", PDF.stat().st_size)


if __name__ == "__main__":
    main()
