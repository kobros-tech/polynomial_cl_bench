# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT
"""Sphinx configuration of the research paper (canonical source: ``index.rst``).

The paper is a Sphinx project of its own, separate from ``docs/``: it builds as a single
article, not as part of the software manual. Build everything with
``python tools/build_paper.py`` (see ``paper/README.rst``); the targets below also work
directly, after ``pip install -e ".[docs]"``::

    sphinx-build -W --keep-going -b html paper paper/_build/html
    sphinx-build -W --keep-going -b latex paper paper/_build/latex && make -C paper/_build/latex
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "tools"))

import paper_rst_assets  # noqa: E402

TITLE = "Does Incremental Polynomial Feature Expansion Help Class-Incremental Learning?"

project = TITLE
author = "Mohamed Alkobrosli"
copyright = "2026, Mohamed Alkobrosli"  # noqa: A001 - Sphinx's required name
release = ""
version = ""

extensions = ["sphinx.ext.mathjax"]
root_doc = "index"
# generated_rst/ holds fragments that index.rst includes; they are not documents.
exclude_patterns = ["_build", "generated", "generated_rst", "README.rst", "legacy"]
language = "en"
smartquotes = False  # the source already uses typographic quotes and apostrophes

numfig = True
numfig_secnum_depth = 0  # number floats and equations sequentially (1, 2, 3, ...)
numfig_format = {"figure": "Fig. %s", "table": "Table %s"}
math_number_all = False
math_eqref_format = "({number})"

# Figures are given without an extension: HTML takes the SVG, LaTeX the PDF.
html_theme = "furo"
html_title = TITLE
html_static_path = []

latex_engine = "pdflatex"
latex_documents = [
    (
        "index",
        "paper.tex",
        # The subtitle is the first paragraph of index.rst, which follows the title block.
        TITLE,
        "Mohamed Alkobrosli",
        "howto",
    ),
]
today = "Draft manuscript"  # the date line of the title block, as in the original draft
latex_elements = {
    "tableofcontents": "",  # the original article had none
    "papersize": "a4paper",
    "pointsize": "11pt",
    "releasename": "",
    # Computer Modern, as the original LaTeX article used, instead of Sphinx's default TeX Gyre
    # fonts: fewer TeX packages are needed to build the PDF.
    "fontpkg": "",
    "sphinxsetup": "hmargin=2.3cm, vmargin=2.3cm",
}
latex_table_style = ["booktabs"]
latex_show_urls = "no"


def _derive_assets(app):
    """Regenerate RST tables/numbers and the SVG/PNG figures from the committed sources."""
    paper_rst_assets.ensure_assets(_HERE)


def setup(app):
    app.connect("builder-inited", _derive_assets)
    return {"parallel_read_safe": True}
