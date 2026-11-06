# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT
"""The paper's reStructuredText source, its derived assets and its three output formats.

Three groups of guarantees:

* the converter that derives RST tables and number substitutions from the generated LaTeX is
  faithful (no value is lost or altered, unknown LaTeX is an error);
* the RST manuscript preserves the content of the original ``paper/main.tex`` (every number
  macro, citation, label, figure and bibliography entry);
* the HTML, PDF and Word builds work and contain what the source contains.

Builds that need external tools (Sphinx, TeX, pandoc, pdftocairo) skip when the tool is
missing; the preservation tests need only the standard library.
"""

from __future__ import annotations

import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
sys.path.insert(0, str(ROOT / "tools"))

import paper_rst_assets as pra  # noqa: E402

INDEX = (PAPER / "index.rst").read_text(encoding="utf-8")
MAIN_TEX = (PAPER / "main.tex").read_text(encoding="utf-8")
GENERATED = sorted((PAPER / "generated").glob("table_*.tex"))


# ------------------------------------------------------------------ the converter


@pytest.mark.parametrize(
    ("tex", "rst"),
    [
        (r"\texttt{frozen\_old}", "``frozen_old``"),
        (r"77.8{\scriptsize$\pm$0.4}", "77.8\u00b10.4"),
        (r"Acc.\ (\%)", "Acc. (%)"),
        (r"9{,}690", "9,690"),
        (r"0--3", "0\u20133"),
        (r"$\eta=[2.5{,} 10]$", ":math:`\\eta=[2.5{,} 10]`"),
        (r"$t_{\rm log}$", r":math:`t_{\mathrm{log}}`"),
        (r"\emph{Ranked}", "*Ranked*"),
        (r"$\Delta$ acc.", ":math:`\\Delta` acc."),
        ("", ""),
    ],
)
def test_inline_conversion(tex, rst):
    assert pra.tex_inline_to_rst(tex) == rst


def test_emphasis_is_closed_around_math():
    # RST cannot nest inline markup; a role inside *...* would print literally.
    rst = pra.tex_inline_to_rst(r"\emph{degrees $1,2,3$)}")
    assert rst == r"*degrees* :math:`1,2,3`\ *)*"


@pytest.mark.parametrize("bad", [r"\textsc{x}", r"\foo", r"$x", r"{unbalanced", "}"])
def test_unknown_or_malformed_latex_is_an_error_not_dropped(bad):
    with pytest.raises(pra.ConversionError):
        pra.tex_inline_to_rst(bad)


def test_numbers_must_be_newcommand_lines_with_unique_names():
    with pytest.raises(pra.ConversionError):
        pra.convert_numbers(
            r"\newcommand{\numA}{1}" + "\n" + r"\renewcommand{\numB}{2}"
        )
    with pytest.raises(pra.ConversionError):
        pra.convert_numbers(r"\newcommand{\numA}{1}" + "\n" + r"\newcommand{\numA}{2}")
    with pytest.raises(
        pra.ConversionError
    ):  # docutils matches substitutions case-blind
        pra.convert_numbers(
            r"\newcommand{\numAb}{1}" + "\n" + r"\newcommand{\numAB}{2}"
        )


def test_row_with_too_many_columns_is_an_error():
    tex = "\\begin{tabular}{ll}\na & b & c \\\\\n\\end{tabular}\n"
    with pytest.raises(pra.ConversionError):
        pra.convert_table(tex)


def test_short_rows_are_padded_as_latex_does():
    tex = "\\begin{tabular}{lll}\n\\toprule\nA & B & C \\\\\n\\midrule\nx & y \\\\\n\\bottomrule\n\\end{tabular}\n"
    assert pra.convert_table(tex) == (
        "+---+---+---+\n| A | B | C |\n+===+===+===+\n| x | y |   |\n+---+---+---+\n"
    )


def _tex_numbers(tex: str) -> list[str]:
    """Digit runs in a generated tabular, without layout-only constructs."""
    tex = re.sub(r"\\begin\{tabular\}\{[^\n]*\}", "", tex)
    tex = re.sub(r"\\multicolumn\{\d+\}\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", "", tex)
    tex = re.sub(r"\\cmidrule(\([^)]*\))?\{[^}]*\}", "", tex)
    tex = re.sub(r"\\addlinespace(\[[^\]]*\])?", "", tex)
    tex = re.sub(r"\\[A-Za-z]+", "", tex)
    return re.findall(r"\d+", tex)


@pytest.fixture(scope="module")
def derived(tmp_path_factory):
    out = tmp_path_factory.mktemp("rst_assets")
    pra.convert_assets(PAPER, out)
    return out


@pytest.mark.parametrize("src", GENERATED, ids=lambda p: p.stem)
def test_every_digit_of_every_generated_table_survives(src, derived):
    rst = (derived / (src.stem + ".rst")).read_text(encoding="utf-8")
    rst_digits = re.findall(r"\d+", rst)
    assert Counter(_tex_numbers(src.read_text(encoding="utf-8"))) == Counter(rst_digits)


@pytest.mark.parametrize("src", GENERATED, ids=lambda p: p.stem)
def test_generated_tables_parse_as_one_table_with_the_source_row_count(src, derived):
    docutils_core = pytest.importorskip("docutils.core")
    docutils_nodes = pytest.importorskip("docutils.nodes")
    rst = (derived / (src.stem + ".rst")).read_text(encoding="utf-8")
    messages = []
    doctree = docutils_core.publish_doctree(
        rst, settings_overrides={"report_level": 2, "warning_stream": messages}
    )
    assert not [m for m in messages]
    tables = list(doctree.findall(docutils_nodes.table))
    assert len(tables) == 1
    rows = list(tables[0].findall(docutils_nodes.row))
    tex = src.read_text(encoding="utf-8")
    n_tex_rows = tex.count("\\\\") - tex.count("\\\\[")  # row terminators
    assert len(rows) == n_tex_rows


def test_number_substitutions_equal_the_generated_macros(derived):
    tex = (PAPER / "generated" / "numbers.tex").read_text(encoding="utf-8")
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", tex, flags=re.M))
    subs = dict(
        re.findall(
            r"^\.\. \|(\w+)\| replace:: (.*)$",
            (derived / "numbers.rst").read_text(),
            flags=re.M,
        )
    )
    assert len(macros) == len(subs) == 352
    assert set(macros) == set(subs)
    for name, raw in macros.items():
        expected = raw.replace("{,}", ",")
        expected = re.sub(
            r"\\texttt\{(.*)\}",
            lambda m: "``" + m.group(1).replace("\\_", "_") + "``",
            expected,
        )
        assert subs[name] == expected, name


# ------------------------------------------------------------------ preservation of main.tex


def _bibliography_tex():
    body = MAIN_TEX.split(r"\begin{thebibliography}{99}")[1].split(
        r"\end{thebibliography}"
    )[0]
    entries = {}
    for m in re.finditer(
        r"\\bibitem\{(\w+)\}\s*(.*?)(?=\\bibitem|\Z)", body, flags=re.S
    ):
        entries[m.group(1)] = m.group(2)
    return entries


def _norm_entry(text: str) -> str:
    text = text.replace("\\'i", "\u00ed").replace("\\'a", "\u00e1")
    text = re.sub(r"\\emph\{([^}]*)\}", r"\1", text)  # TeX \emph{x}
    text = re.sub(r"\*([^*]+)\*", r"\1", text)  # RST *x*
    text = text.replace("~", " ").replace("\\ ", " ").replace("--", "\u2013")
    text = text.replace("\\", "").replace("\u2019", "'").replace("'", "'")
    return re.sub(r"\s+", " ", text).strip()


def _bibliography_rst():
    entries = {}
    for m in re.finditer(r"^\.\. \[(\w+)\] (.*)$", INDEX, flags=re.M):
        entries[m.group(1)] = m.group(2)
    return entries


def test_bibliography_is_preserved_entry_by_entry():
    tex, rst = _bibliography_tex(), _bibliography_rst()
    assert len(tex) == 20
    assert set(tex) == set(rst)
    for key in tex:
        assert _norm_entry(rst[key]) == _norm_entry(tex[key]), key


def test_every_citation_is_kept_with_its_multiplicity():
    tex_body = MAIN_TEX.split(r"\begin{thebibliography}")[0]
    tex_cites = Counter(
        k.strip()
        for group in re.findall(r"\\cite\{([^}]*)\}", tex_body)
        for k in group.split(",")
    )
    rst_body = INDEX.split("\nReferences\n==========")[0]
    rst_cites = Counter(re.findall(r"\[(\w+)\]_", rst_body))
    assert tex_cites == rst_cites
    assert set(rst_cites) == set(_bibliography_rst())  # no entry uncited, none missing


def test_every_number_macro_is_used_exactly_as_in_the_original():
    tex_body = MAIN_TEX.split(r"\input{generated/numbers.tex}")[1]
    tex_used = Counter(re.findall(r"\\(num[A-Za-z]+)", tex_body))
    rst_used = Counter(re.findall(r"\|(num[A-Za-z]+)\|", INDEX))
    assert sum(tex_used.values()) >= 100  # guards the test itself
    assert tex_used == rst_used


def test_cross_references_point_at_the_same_floats():
    tex_refs = Counter(re.findall(r"\\ref\{((?:tab|fig):\w+)\}", MAIN_TEX))
    rst_refs = Counter(re.findall(r":numref:`((?:tab|fig)-\w+)`", INDEX))
    assert {k.replace(":", "-"): v for k, v in tex_refs.items()} == dict(rst_refs)
    tex_labels = set(re.findall(r"\\label\{((?:tab|fig):\w+)\}", MAIN_TEX))
    rst_labels = set(re.findall(r"^\.\. _((?:tab|fig)-\w+):$", INDEX, flags=re.M))
    assert {k.replace(":", "-") for k in tex_labels} == rst_labels


def test_figures_tables_and_equations_are_all_present():
    assert Counter(re.findall(r"figures/(\w+)\.pdf", MAIN_TEX)) == Counter(
        re.findall(r"figures/(\w+)\.\*", INDEX)
    )
    tex_tables = re.findall(r"input\{generated/(table_\w+)\.tex\}", MAIN_TEX)
    rst_tables = re.findall(r"include:: generated_rst/(table_\w+)\.rst", INDEX)
    assert sorted(tex_tables) == sorted(rst_tables) == sorted(p.stem for p in GENERATED)
    assert MAIN_TEX.count(r"\begin{equation}") == INDEX.count(".. math::") == 2
    for name in re.findall(r"figures/(\w+)\.\*", INDEX):
        assert (PAPER / "figures" / f"{name}.pdf").is_file()


def _prose_figures(text: str) -> list[str]:
    """Percentages and decimals typed into running text (not generated, not in math/code)."""
    return sorted(
        x.replace("\\ ", "")
        for x in re.findall(r"\d+(?:\.\d+)?(?:\\ )?%|\d+\.\d+", text)
    )


def test_no_hand_typed_result_was_added_during_migration():
    """The draft's prose contains a few hand-typed figures (listed in paper/README.rst, to be
    validated or generated in PR 5). The migration must neither add nor drop one."""
    rst = INDEX.split("\nIntroduction\n")[1].split("\nAppendix: Reproducibility\n")[0]
    rst = re.sub(
        r"^\.\. (table|figure|list-table|math)::.*?(?=\n\S|\Z)",
        "",
        rst,
        flags=re.S | re.M,
    )
    for pattern in (r":math:`[^`]*`", r"\|num\w+\|", r"\[\w+\]_", r"``[^`]*``"):
        rst = re.sub(pattern, "", rst)
    tex = MAIN_TEX.split(r"\section{Introduction}")[1].split(r"\appendix")[0]
    tex = re.sub(
        r"\\begin\{(table|figure|equation)\}.*?\\end\{\1\}", "", tex, flags=re.S
    )
    tex = re.sub(r"\$[^$]*\$", "", tex)
    tex = re.sub(r"\\num\w+(\\%)?|\\cite\{[^}]*\}|\\texttt\{[^}]*\}", "", tex)
    tex = re.sub(r"\[leftmargin=[^\]]*\]", "", tex)  # a layout length, not a result
    tex = tex.replace("\\%", "%")
    assert _prose_figures(rst) == _prose_figures(tex)
    assert len(_prose_figures(rst)) == 13


# ------------------------------------------------------------------ HTML build


needs_sphinx = pytest.mark.skipif(
    shutil.which("pdftocairo") is None, reason="pdftocairo (poppler-utils) missing"
)


@pytest.fixture(scope="module")
def html_dir(tmp_path_factory):
    pytest.importorskip("sphinx")
    pytest.importorskip("furo")
    if shutil.which("pdftocairo") is None:
        pytest.skip("pdftocairo (poppler-utils) missing")
    import subprocess

    out = tmp_path_factory.mktemp("html")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-W",
            "--keep-going",
            "-b",
            "html",
            str(PAPER),
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return out


@needs_sphinx
def test_html_builds_without_warnings_and_has_no_markup_leftovers(html_dir):
    html = (html_dir / "index.html").read_text(encoding="utf-8")
    text = re.sub(r"<[^>]+>", " ", html)
    for leftover in (":math:", "|num", "??", "numref", "Failed to create"):
        assert leftover not in text, leftover
    assert html.count('class="math') >= 100  # inline and display math reached MathJax
    assert len(re.findall(r"<table\b", html)) == 6
    assert "Table 6" in text and "Fig. 3" in text  # sequential numbering, not 5.1
    assert len(re.findall(r'class="citation"', html)) == 20


@needs_sphinx
def test_html_figures_are_the_derived_svgs_and_exist(html_dir):
    html = (html_dir / "index.html").read_text(encoding="utf-8")
    images = re.findall(r'src="(_images/[^"]+)"', html)
    assert len(images) == 3 and all(i.endswith(".svg") for i in images)
    assert all((html_dir / i).is_file() for i in images)


# ------------------------------------------------------------------ Word build


def _docx_paragraphs(xml: str) -> list[str]:
    """Text of each Word paragraph, read through an XML parser.

    Different pandoc versions write ``document.xml`` compact or pretty-printed; the
    indentation of the latter must not matter, so text is taken from ``w:t``/``m:t`` elements
    only and never from the whitespace between tags.
    """
    from xml.etree import ElementTree

    root = ElementTree.fromstring(xml)
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs = []
    for para in root.iter(ns + "p"):
        text = "".join(
            el.text or ""
            for el in para.iter()
            if el.tag.endswith("}t") and el.text is not None
        )
        paragraphs.append(text.strip())
    return paragraphs


@pytest.fixture(scope="module")
def docx_parts():
    if shutil.which("pandoc") is None or shutil.which("pdftocairo") is None:
        pytest.skip("pandoc or pdftocairo missing")
    import build_paper

    pra.ensure_assets(PAPER)
    path = build_paper.build_docx()
    with zipfile.ZipFile(path) as z:
        return z.read("word/document.xml").decode("utf-8"), sorted(
            n for n in z.namelist() if n.startswith("word/media/")
        )


def test_docx_has_the_equations_tables_figures_and_numbers(docx_parts):
    xml, media = docx_parts
    assert xml.count("<w:tbl>") == 6
    assert (
        xml.count("<m:oMathPara>") == 2
    )  # the two display equations, as Word equations
    assert xml.count("<m:oMath>") >= 100
    assert xml.count("<w:drawing>") == len(media) == 3
    assert all(m.endswith(".png") for m in media)
    assert xml.count("<w:gridSpan") == 6  # the column spans of the generated tables


def test_docx_cross_references_captions_and_citations_are_resolved(docx_parts):
    xml, _ = docx_parts
    paragraphs = _docx_paragraphs(xml)
    text = "\n".join(paragraphs)
    for leftover in ("|num", "numref", "??", "tab-", "fig-"):
        assert leftover not in text, leftover
    assert sorted(re.findall(r"Table (\d)\.", text)) == list("123456")
    assert sorted(re.findall(r"Fig\. (\d)\.", text)) == list("123")
    assert "Table 2 gives" in text and "Table 6" in text
    for key in _bibliography_rst():
        assert f"[{key}]" in text, key  # cited in the running text
        assert key in paragraphs, key  # and listed in the bibliography
    assert "77.8%" in text and "13,834" in text  # generated values reached the document
    # Known limitation (docs): the Word equations carry no numbers.
    assert "(1)" not in text


def test_docx_text_extraction_ignores_xml_pretty_printing(docx_parts):
    """The check above must give the same answer for compact and indented XML."""
    from xml.dom import minidom

    xml, _ = docx_parts
    pretty = minidom.parseString(xml.encode("utf-8")).toprettyxml(indent="    ")
    assert pretty.count("\n") > xml.count("\n")  # really re-indented
    assert _docx_paragraphs(pretty) == _docx_paragraphs(xml)
