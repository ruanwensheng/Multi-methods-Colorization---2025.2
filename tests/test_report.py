"""Structural guards for the LaTeX report at reports/deep_learning/.

No LaTeX engine is installed in CI, so these tests stand in for the compiler:
they verify the report's structural integrity (IEEEtran class, section wiring,
required BibTeX keys, required figures) and run a consistency lint that catches
the two errors pdflatex/bibtex would otherwise surface late — dangling \\cite
keys and \\includegraphics paths that don't exist. Content-depth guards (per
section) live alongside so the report can't regress to empty stubs.
"""

import os
import re

REPORT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "reports", "deep_learning"))
SECTIONS_DIR = os.path.join(REPORT, "sections")
MAIN = os.path.join(REPORT, "main.tex")
BIB = os.path.join(REPORT, "references.bib")
FIGURES_DIR = os.path.join(REPORT, "figures")

SECTION_NAMES = ["introduction", "related_work", "method",
                 "experiments", "results", "discussion", "conclusion"]

# Spec section 8: the 8 BibTeX entries the report must cite.
REQUIRED_BIB_KEYS = [
    "zhang2016colorful", "zhang2017realtime", "vitoria2020chromagan",
    "antic2019deoldify", "saharia2022palette", "zhang2023controlnet",
    "lin2014coco", "zhang2018lpips",
]

# Spec section 8: the 5 figures referenced by the report.
REQUIRED_FIGURES = [
    "architecture.png", "ab_quantization.png", "training_curves.png",
    "metrics_comparison.png", "qualitative_grid.png",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _section_path(name):
    return os.path.join(SECTIONS_DIR, name + ".tex")


def _all_tex_text():
    """Concatenate main.tex + every section file that exists."""
    parts = [_read(MAIN)] if os.path.exists(MAIN) else []
    for name in SECTION_NAMES:
        p = _section_path(name)
        if os.path.exists(p):
            parts.append(_read(p))
    return "\n".join(parts)


def _strip_comments(text):
    """Drop LaTeX line comments so commented-out \\cite/\\includegraphics don't count."""
    out = []
    for line in text.splitlines():
        m = re.search(r"(?<!\\)%", line)
        out.append(line[: m.start()] if m else line)
    return "\n".join(out)


def _bib_keys():
    if not os.path.exists(BIB):
        return set()
    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", _read(BIB)))


def _cite_keys():
    text = _strip_comments(_all_tex_text())
    keys = set()
    for grp in re.findall(r"\\cite[a-z]*\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}", text):
        keys.update(k.strip() for k in grp.split(",") if k.strip())
    return keys


def _includegraphics_targets():
    text = _strip_comments(_all_tex_text())
    return re.findall(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}", text)


def _graphics_search_dirs():
    """Dirs pdflatex searches for figures: REPORT plus any \\graphicspath entry."""
    dirs = [REPORT]
    if os.path.exists(MAIN):
        for block in re.findall(r"\\graphicspath\s*\{(.+?)\}\s*$", _read(MAIN), re.MULTILINE):
            for d in re.findall(r"\{([^}]*)\}", block):
                dirs.append(os.path.normpath(os.path.join(REPORT, d)))
    return dirs


# --------------------------------------------------------------------------- #
# T5.1 — skeleton
# --------------------------------------------------------------------------- #
def test_main_tex_exists_and_uses_ieeetran():
    assert os.path.exists(MAIN), "reports/deep_learning/main.tex missing"
    text = _read(MAIN)
    assert re.search(r"\\documentclass(\[[^\]]*\])?\{IEEEtran\}", text), \
        "main.tex must use the IEEEtran document class"
    assert "\\begin{document}" in text and "\\end{document}" in text


def test_main_inputs_all_sections():
    text = _strip_comments(_read(MAIN))
    inputs = set(re.findall(r"\\(?:input|include)\s*\{([^}]*)\}", text))
    inputs = {os.path.basename(p).replace(".tex", "") for p in inputs}
    for name in SECTION_NAMES:
        assert name in inputs, f"main.tex does not \\input sections/{name}"


def test_all_section_files_exist():
    for name in SECTION_NAMES:
        assert os.path.exists(_section_path(name)), f"missing sections/{name}.tex"


def test_references_bib_has_all_required_keys():
    keys = _bib_keys()
    missing = [k for k in REQUIRED_BIB_KEYS if k not in keys]
    assert not missing, f"references.bib missing required entries: {missing}"


def test_required_figures_present():
    missing = [f for f in REQUIRED_FIGURES
               if not os.path.exists(os.path.join(FIGURES_DIR, f))]
    assert not missing, f"missing figures: {missing}"


def test_main_declares_bibliography():
    text = _read(MAIN)
    assert re.search(r"\\bibliography\s*\{", text), \
        "main.tex must \\bibliography{references} for BibTeX"


# --------------------------------------------------------------------------- #
# build-gate surrogate — what pdflatex/bibtex would catch
# --------------------------------------------------------------------------- #
def test_every_citation_resolves_to_a_bib_entry():
    """No dangling \\cite — bibtex would emit 'undefined citation' otherwise."""
    dangling = sorted(_cite_keys() - _bib_keys())
    assert not dangling, f"\\cite keys with no references.bib entry: {dangling}"


def test_every_includegraphics_file_exists():
    """No \\includegraphics to a missing file — pdflatex would error out.

    Resolves each target against the same search path pdflatex uses (REPORT plus
    every \\graphicspath dir), trying common extensions when none is given.
    """
    search_dirs = _graphics_search_dirs()
    missing = []
    for target in _includegraphics_targets():
        cands = [target] if target.lower().endswith((".png", ".jpg", ".pdf", ".eps")) \
            else [target + ext for ext in (".png", ".pdf", ".jpg", ".eps")]
        found = any(os.path.exists(os.path.join(d, c)) for d in search_dirs for c in cands)
        if not found:
            missing.append(target)
    assert not missing, f"\\includegraphics targets not found in {search_dirs}: {missing}"


# --------------------------------------------------------------------------- #
# content-depth guards (T5.2–T5.5) — keep sections from regressing to stubs
# --------------------------------------------------------------------------- #
def test_each_section_has_substantive_content():
    for name in SECTION_NAMES:
        p = _section_path(name)
        if not os.path.exists(p):
            continue
        body = _strip_comments(_read(p)).strip()
        # A real section is well past stub length and opens with \section.
        assert "\\section" in body, f"{name}.tex has no \\section heading"
        assert len(body) > 400, f"{name}.tex looks like a stub ({len(body)} chars)"


def test_results_section_reports_real_benchmark_numbers():
    """results.tex must carry the canonical headline numbers, not placeholders."""
    body = _read(_section_path("results")) if os.path.exists(_section_path("results")) else ""
    for token in ["23.29", "24.00", "18.82", "17.14"]:
        assert token in body, f"results.tex missing benchmark PSNR {token}"


def test_related_work_cites_the_four_paradigms():
    """related_work.tex must cite the CNN/Interactive/GAN/Diffusion sources."""
    body = _read(_section_path("related_work")) if os.path.exists(_section_path("related_work")) else ""
    for key in ["zhang2016colorful", "zhang2017realtime", "vitoria2020chromagan", "saharia2022palette"]:
        assert key in body, f"related_work.tex does not cite {key}"
