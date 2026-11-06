# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT
"""Derive the paper's reStructuredText assets from the generated LaTeX fragments.

``tools/build_paper_assets.py`` writes the tables and number macros of the paper as LaTeX
(``paper/generated/*.tex``). The manuscript is now written in reStructuredText (``paper/``),
so this module converts those generated fragments, **mechanically and without retyping any
value**, into

* ``numbers.rst``     one ``.. |numName| replace:: value`` substitution per ``\\newcommand``;
* ``table_*.rst``     one RST grid table per ``tabular`` (grid tables, because the tables use
  column spans); and into the web/Word renderings of the PDF figures
  (``figures/*.svg`` and ``figures/*.png``, via ``pdftocairo``).

The converter is *strict*: any LaTeX command it does not know is an error, never silently
dropped, so a change in the generator cannot lose content unnoticed. Purely presentational
commands (rules, ``\\scriptsize``, ``\\addlinespace``) are dropped on purpose and listed in
``PRESENTATION_ONLY``. Standard library only.

    python tools/paper_rst_assets.py --paper-dir paper          # tables, numbers, figures
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

NBSP = "\u00a0"

# Commands that only affect typesetting; dropping them cannot change a value or a word.
PRESENTATION_ONLY = {
    "toprule",
    "bottomrule",
    "midrule",
    "scriptsize",
    "small",
    "footnotesize",
    "addlinespace",
    "cmidrule",
}

# A `$...$` whose whole content is one of these is written as plain text, not as a math role.
MATH_AS_TEXT = {r"\pm": "\u00b1"}  # \Delta stays math: pdfLaTeX cannot typeset U+0394

TEX_ESCAPES = {
    "_": "_",
    "%": "%",
    "&": "&",
    "#": "#",
    "$": "$",
    "{": "{",
    "}": "}",
    " ": " ",
    ",": "\u2009",
    "\\": "\\",
}


class ConversionError(ValueError):
    """The input contains LaTeX that this converter does not know how to translate."""


# ------------------------------------------------------------------ brace-aware scanning


def _matching_brace(text: str, start: int) -> int:
    """Index of the ``}`` that closes the ``{`` at ``start``."""
    depth = 0
    i = start
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ConversionError(f"unbalanced braces in {text!r}")


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split on ``sep`` outside braces and not preceded by a backslash."""
    parts, depth, buf, i = [], 0, [], 0
    while i < len(text):
        c = text[i]
        if c == "\\" and i + 1 < len(text):
            if text.startswith(sep, i) and depth == 0 and sep.startswith("\\"):
                parts.append("".join(buf))
                buf = []
                i += len(sep)
                continue
            buf.append(text[i : i + 2])
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        if depth == 0 and not sep.startswith("\\") and text.startswith(sep, i):
            parts.append("".join(buf))
            buf = []
            i += len(sep)
            continue
        buf.append(c)
        i += 1
    parts.append("".join(buf))
    return parts


# ------------------------------------------------------------------ inline conversion


def _rst_escape(text: str) -> str:
    """Escape characters that RST would read as markup in running text."""
    out = []
    for ch in text:
        if ch in "\\*`|_":
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def _math_to_rst(content: str) -> str:
    content = content.strip()
    if content in MATH_AS_TEXT:
        return MATH_AS_TEXT[content]
    # `\rm` is a deprecated switch; MathJax and LaTeX both accept `\mathrm{...}`.
    content = re.sub(r"\\rm\s+([A-Za-z]+)\}", r"\\mathrm{\1}}", content)
    if "\\rm" in content:
        raise ConversionError(f"unhandled \\rm in math: {content!r}")
    if "`" in content:
        raise ConversionError(f"backtick in math: {content!r}")
    return f":math:`{content}`"


_MATH_ROLE = re.compile(r":math:`[^`]*`")


def _emphasise(rst: str) -> str:
    """``*...*`` around already-converted text. RST cannot nest inline markup, so a math role
    inside emphasis would print literally; the emphasis is closed around each role instead."""
    if ":math:" not in rst:
        return f"*{rst}*"
    out, pos = [], 0
    for m in _MATH_ROLE.finditer(rst):
        out.append((True, rst[pos : m.start()]))
        out.append((False, m.group(0)))
        pos = m.end()
    out.append((True, rst[pos:]))
    pieces = []
    for is_text, chunk in out:
        if not is_text:
            pieces.append(chunk)
            continue
        core = chunk.strip(" ")
        if not core:
            pieces.append(chunk)
            continue
        lead = " " if chunk.startswith(" ") else "\\ " if pieces else ""
        trail = " " if chunk.endswith(" ") else "\\ "
        pieces.append(f"{lead}*{core}*{trail}")
    result = "".join(pieces)
    return result[: -len("\\ ")] if result.endswith("\\ ") else result


def tex_inline_to_rst(text: str) -> str:
    """Convert one run of LaTeX text (a table cell or a macro value) to RST inline text."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "$":
            j = text.find("$", i + 1)
            if j < 0:
                raise ConversionError(f"unterminated math in {text!r}")
            out.append(_math_to_rst(text[i + 1 : j]))
            i = j + 1
        elif c == "\\":
            m = re.match(r"\\([A-Za-z]+)", text[i:])
            if m is None:  # control symbol: \_ \% \  \, ...
                nxt = text[i + 1] if i + 1 < n else ""
                if nxt not in TEX_ESCAPES:
                    raise ConversionError(f"unknown control symbol \\{nxt} in {text!r}")
                out.append(
                    _rst_escape(TEX_ESCAPES[nxt]) if nxt in "\\" else TEX_ESCAPES[nxt]
                )
                i += 2
                continue
            name = m.group(1)
            i += len(m.group(0))
            if name in PRESENTATION_ONLY:
                continue
            if name in {"texttt", "emph", "textbf"}:
                if i >= n or text[i] != "{":
                    raise ConversionError(f"\\{name} without argument in {text!r}")
                end = _matching_brace(text, i)
                inner = text[i + 1 : end]
                i = end + 1
                if name == "texttt":
                    literal = tex_inline_to_rst(inner)
                    # A literal must not contain markup: undo escapes and drop roles.
                    literal = literal.replace("\\_", "_").replace("\\*", "*")
                    if ":math:" in literal or "``" in literal:
                        raise ConversionError(f"markup inside texttt: {inner!r}")
                    out.append(f"``{literal}``")
                elif name == "emph":
                    out.append(_emphasise(tex_inline_to_rst(inner)))
                else:
                    out.append(f"**{tex_inline_to_rst(inner)}**")
            else:
                raise ConversionError(f"unknown LaTeX command \\{name} in {text!r}")
        elif c == "{":
            end = _matching_brace(text, i)
            inner = text[i + 1 : end]
            i = end + 1
            # `{,}` is the thousands/list separator protected from math spacing.
            out.append("," if inner == "," else tex_inline_to_rst(inner))
        elif c == "}":
            raise ConversionError(f"unbalanced '}}' in {text!r}")
        elif text.startswith("--", i):
            out.append("\u2013")
            i += 2
        elif c == "~":
            out.append(NBSP)
            i += 1
        else:
            out.append(_rst_escape(c))
            i += 1
    return "".join(out)


# ------------------------------------------------------------------ tables


def _parse_tabular(tex: str):
    """Return ``(ncols, header_rows, body_rows)`` where a row is ``[(raw_cell, span), ...]``."""
    m = re.fullmatch(
        r"\s*\\begin\{tabular\}\{([^\n]*?)\}\s*\n(.*?)\\end\{tabular\}\s*",
        tex,
        flags=re.S,
    )
    if m is None:
        raise ConversionError("input is not a single tabular environment")
    spec = re.sub(r"@\{[^}]*\}", "", m.group(1))
    ncols = len(re.findall(r"[lrcp]", spec))
    body = m.group(2)

    rows, header_rows, seen_midrule = [], [], False
    # Join physical lines, then split into logical rows on the row terminator.
    for chunk in _split_top_level(body, r"\\"):
        chunk = chunk.strip()
        if not chunk:
            continue
        # Rules and spacing commands sit between rows, on lines of their own.
        lines = []
        for line in chunk.split("\n"):
            stripped = line.strip()
            if re.fullmatch(
                r"(\\(toprule|bottomrule|midrule|addlinespace(\[[^\]]*\])?|cmidrule(\([^)]*\))?\{[^}]*\}))*",
                stripped,
            ):
                if "\\midrule" in stripped and not seen_midrule:
                    seen_midrule = True
                    header_rows = rows
                    rows = []
                continue
            lines.append(stripped)
        chunk = " ".join(lines).strip()
        if not chunk:
            continue
        cells, col = [], 0
        for raw in _split_top_level(chunk, "&"):
            raw = raw.strip()
            span = 1
            if raw.startswith("\\multicolumn"):
                # \multicolumn{span}{column spec}{content}; the spec may itself contain braces.
                pos = len("\\multicolumn")
                args = []
                for _ in range(3):
                    if pos >= len(raw) or raw[pos] != "{":
                        raise ConversionError(f"malformed \\multicolumn: {raw!r}")
                    end = _matching_brace(raw, pos)
                    args.append(raw[pos + 1 : end])
                    pos = end + 1
                if raw[pos:].strip():
                    raise ConversionError(f"text after \\multicolumn: {raw!r}")
                span, raw = int(args[0]), args[2].strip()
            cells.append((raw, span))
            col += span
        if col > ncols:
            # LaTeX tolerates a stray trailing `&` only if the extra cells are empty.
            while col > ncols and cells and cells[-1] == ("", 1):
                cells.pop()
                col -= 1
            if col > ncols:
                raise ConversionError(
                    f"row has {col} columns, expected {ncols}: {chunk[:80]!r}"
                )
        # LaTeX also tolerates rows with *fewer* cells than the column specification.
        rows.append(cells)
    if not seen_midrule:
        header_rows = []
    # Effective width: up to the last non-empty cell. The generated column specification can
    # declare more columns than any row uses (table_cells.tex declares 11, uses 10); trailing
    # empty columns carry no content, so they are not reproduced.
    used = 0
    for row in (*header_rows, *rows):
        col = 0
        for raw, span in row:
            col += span
            if raw:
                used = max(used, col)
    ncols = min(ncols, used)
    for row in (*header_rows, *rows):
        col = sum(span for _raw, span in row)
        while col > ncols and row and row[-1] == ("", 1):
            row.pop()
            col -= 1
        if col > ncols:
            raise ConversionError("a cell spans past the last used column")
        row.extend([("", 1)] * (ncols - col))
    return ncols, header_rows, rows


def _grid_table(ncols: int, header, body) -> str:
    """Render rows of ``(rst_text, span)`` as a grid table (spans are column spans)."""
    all_rows = [*header, *body]
    width = [3] * ncols
    for row in all_rows:
        col = 0
        for text, span in row:
            if span == 1:
                width[col] = max(width[col], len(text) + 2)
            col += span
    for row in all_rows:
        col = 0
        for text, span in row:
            if span > 1:
                avail = sum(width[col : col + span]) + (span - 1)
                if avail < len(text) + 2:
                    width[col + span - 1] += len(text) + 2 - avail
            col += span

    def boundaries(row):
        """Column boundaries (0..ncols) at which this row has a cell edge."""
        edges, col = {0}, 0
        for _text, span in row:
            col += span
            edges.add(col)
        return edges

    def border(above, below, char):
        edges = set()
        for row in (above, below):
            if row is not None:
                edges |= boundaries(row)
        parts = ["+"]
        for c in range(ncols):
            parts.append(char * width[c])
            parts.append("+" if (c + 1) in edges else char)
        return "".join(parts)

    def line(row):
        parts, col = ["|"], 0
        for text, span in row:
            w = sum(width[col : col + span]) + (span - 1)
            parts.append(" " + text.ljust(w - 1) + "|")
            col += span
        return "".join(parts)

    out = []
    prev = None
    for idx, row in enumerate(all_rows):
        sep_char = "=" if header and idx == len(header) else "-"
        out.append(border(prev, row, sep_char))
        out.append(line(row))
        prev = row
    out.append(border(prev, None, "-"))
    return "\n".join(out) + "\n"


def convert_table(tex: str) -> str:
    """Convert one generated ``tabular`` to an RST grid table."""
    ncols, header, body = _parse_tabular(tex)

    def conv(rows):
        return [[(tex_inline_to_rst(raw), span) for raw, span in row] for row in rows]

    return _grid_table(ncols, conv(header), conv(body))


# ------------------------------------------------------------------ number macros

_MACRO = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$")


def parse_numbers(tex: str) -> dict[str, str]:
    """``{macro name: raw LaTeX value}`` from ``numbers.tex``; strict about every line."""
    macros: dict[str, str] = {}
    for lineno, line in enumerate(tex.splitlines(), 1):
        if not line.strip():
            continue
        m = _MACRO.fullmatch(line)
        if m is None:
            raise ConversionError(
                f"numbers.tex line {lineno} is not a \\newcommand: {line!r}"
            )
        if m.group(1) in macros:
            raise ConversionError(f"macro defined twice: {m.group(1)}")
        macros[m.group(1)] = m.group(2)
    lowered = {}
    for name in macros:
        if lowered.setdefault(name.lower(), name) != name:
            # docutils falls back to case-insensitive substitution matching
            raise ConversionError(
                f"macro names differ only by case: {name}, {lowered[name.lower()]}"
            )
    return macros


def convert_numbers(tex: str) -> str:
    macros = parse_numbers(tex)
    lines = [
        ".. Generated by tools/paper_rst_assets.py from generated/numbers.tex.",
        ".. Do not edit: the values come from the benchmark's analysis pipeline.",
        "",
    ]
    for name in sorted(macros):
        lines.append(f".. |{name}| replace:: {tex_inline_to_rst(macros[name])}")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ driver


def convert_assets(paper_dir: Path, out_dir: Path) -> list[Path]:
    """Write ``numbers.rst`` and ``table_*.rst`` into ``out_dir``; return the written files."""
    gen = paper_dir / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    numbers = gen / "numbers.tex"
    if not numbers.is_file():
        raise FileNotFoundError(numbers)
    target = out_dir / "numbers.rst"
    target.write_text(
        convert_numbers(numbers.read_text(encoding="utf-8")), encoding="utf-8"
    )
    written.append(target)
    tables = sorted(gen.glob("table_*.tex"))
    if not tables:
        raise FileNotFoundError(f"no table_*.tex in {gen}")
    for src in tables:
        target = out_dir / (src.stem + ".rst")
        target.write_text(
            convert_table(src.read_text(encoding="utf-8")), encoding="utf-8"
        )
        written.append(target)
    return written


def derive_figures(figures_dir: Path) -> list[Path]:
    """Render each ``fig_*.pdf`` as SVG (HTML) and PNG (Word); keep up-to-date files."""
    exe = shutil.which("pdftocairo")
    if exe is None:
        raise RuntimeError(
            "pdftocairo (poppler-utils) is required to derive the HTML and Word figures "
            "from the PDF figures"
        )
    written = []
    for pdf in sorted(figures_dir.glob("*.pdf")):
        for fmt, args in (
            ("svg", ["-svg"]),
            ("png", ["-png", "-r", "200", "-singlefile"]),
        ):
            target = pdf.with_suffix("." + fmt)
            if target.is_file() and target.stat().st_mtime >= pdf.stat().st_mtime:
                continue
            base = str(target.with_suffix("")) if fmt == "png" else str(target)
            subprocess.run(
                [exe, *args, str(pdf), base],
                check=True,
                capture_output=True,
            )
            written.append(target)
    return written


def ensure_assets(paper_dir: Path) -> None:
    """Idempotent: everything a Sphinx or pandoc build of the paper needs, from the sources."""
    convert_assets(paper_dir, paper_dir / "generated_rst")
    derive_figures(paper_dir / "figures")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paper-dir", type=Path, default=Path("paper"))
    args = ap.parse_args(argv)
    try:
        ensure_assets(args.paper_dir)
    except (ConversionError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"derived RST assets and figures under {args.paper_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
