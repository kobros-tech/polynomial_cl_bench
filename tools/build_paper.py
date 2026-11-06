# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT
"""Build the research paper from its canonical reStructuredText source (``paper/index.rst``).

    python tools/build_paper.py [assets|html|latex|pdf|docx|all]    (default: all)

Every output is produced from the RST source and the generated assets; nothing is converted
from another output (no HTML -> PDF/Word). Outputs go to ``paper/_build/`` (not versioned).

* ``assets``  derive RST tables/numbers and SVG/PNG figures (tools/paper_rst_assets.py);
* ``html``    Sphinx HTML;
* ``latex``   Sphinx LaTeX sources;
* ``pdf``     latex + latexmk (needs a TeX distribution);
* ``docx``    pandoc on the RST source, with ``.. include::`` expanded and
              tools/pandoc_paper.lua resolving Sphinx cross-references (needs pandoc).

Sphinx runs with ``-W``: any unresolved citation, reference or figure fails the build.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
BUILD = PAPER / "_build"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import paper_rst_assets  # noqa: E402

_INCLUDE = re.compile(r"^(?P<indent>[ \t]*)\.\. include:: (?P<path>\S+)\s*$")


def expand_includes(path: Path, _seen: tuple[Path, ...] = ()) -> str:
    """Textually expand ``.. include::`` (nested; indentation kept) for pandoc.

    Pandoc does not see substitution definitions that live in an included file, and the
    generated tables are included into captioned ``table`` directives, so the Word build
    works on the expanded source. Relative paths are resolved against the including file.
    """
    path = path.resolve()
    if path in _seen:
        raise RuntimeError(f"circular include: {path}")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _INCLUDE.match(line)
        if m is None:
            out.append(line)
            continue
        target = (path.parent / m["path"]).resolve()
        for inc_line in expand_includes(target, (*_seen, path)).splitlines():
            out.append(m["indent"] + inc_line if inc_line.strip() else "")
    return "\n".join(out) + "\n"


def _run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def build_html() -> Path:
    _run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-W",
            "--keep-going",
            "-b",
            "html",
            PAPER,
            BUILD / "html",
        ]
    )
    return BUILD / "html" / "index.html"


def build_latex() -> Path:
    _run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-W",
            "--keep-going",
            "-b",
            "latex",
            PAPER,
            BUILD / "latex",
        ]
    )
    return BUILD / "latex" / "paper.tex"


def build_pdf() -> Path:
    build_latex()
    if shutil.which("latexmk") is None:
        raise RuntimeError("latexmk (a TeX distribution) is required to build the PDF")
    _run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "paper.tex"],
        cwd=BUILD / "latex",
        stdout=subprocess.DEVNULL,
    )
    return BUILD / "latex" / "paper.pdf"


def build_docx() -> Path:
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        raise RuntimeError("pandoc is required to build the Word version")
    out_dir = BUILD / "docx"
    out_dir.mkdir(parents=True, exist_ok=True)
    expanded = out_dir / "paper_expanded.rst"
    expanded.write_text(expand_includes(PAPER / "index.rst"), encoding="utf-8")
    target = out_dir / "paper.docx"
    _run(
        [
            pandoc,
            "--from=rst",
            "--to=docx",
            f"--lua-filter={ROOT / 'tools' / 'pandoc_paper.lua'}",
            f"--resource-path={PAPER}",
            "--fail-if-warnings",
            f"--output={target}",
            expanded,
        ]
    )
    return target


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "target",
        nargs="?",
        default="all",
        choices=["assets", "html", "latex", "pdf", "docx", "all"],
    )
    args = ap.parse_args(argv)
    steps = {
        "assets": lambda: paper_rst_assets.ensure_assets(PAPER),
        "html": build_html,
        "latex": build_latex,
        "pdf": build_pdf,
        "docx": build_docx,
    }
    order = (
        ["assets", "html", "pdf", "docx"]
        if args.target == "all"
        else ["assets", args.target]
    )
    try:
        paper_rst_assets.ensure_assets(PAPER)
        for name in (
            order[1:] if order[0] == "assets" and args.target != "assets" else order
        ):
            result = steps[name]()
            if result is not None:
                print(f"{name}: {result}")
    except (
        RuntimeError,
        subprocess.CalledProcessError,
        paper_rst_assets.ConversionError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
