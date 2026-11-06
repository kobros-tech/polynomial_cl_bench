# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Sphinx configuration for the polynomial_cl_bench documentation.

Build (from the repository root, after ``pip install -e ".[docs]"``)::

    sphinx-build -W --keep-going -b html docs docs/_build/html
    sphinx-build -W --keep-going -b latex docs docs/_build/latex
    make -C docs/_build/latex          # PDF; needs a TeX distribution with latexmk
"""

from polynomial_cl_bench import __version__

project = "polynomial_cl_bench"
author = "Mohamed Alkobrosli"
copyright = "2026, Mohamed Alkobrosli"  # noqa: A001 - Sphinx's required name
release = __version__
version = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
]

# Hermetic build: no intersphinx, so nothing is fetched from the network while building.
root_doc = "index"
exclude_patterns = ["_build"]
nitpicky = False

autodoc_member_order = "bysource"
autodoc_default_options = {"show-inheritance": True}

html_theme = "furo"
html_title = f"polynomial_cl_bench {release}"

latex_engine = "pdflatex"
latex_documents = [
    (
        "index",
        "polynomial_cl_bench.tex",
        "polynomial\\_cl\\_bench: documentation",
        "Mohamed Alkobrosli",
        "manual",
    ),
]
latex_elements = {"papersize": "a4paper", "pointsize": "11pt"}
