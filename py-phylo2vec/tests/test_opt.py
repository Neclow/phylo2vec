"""Tests for phylo2vec.opt"""

import subprocess
import sys

# Hide jax, so that the [gradme] extras look missing even if they are installed
HIDE_JAX = "import sys; sys.modules['jax'] = None; "


def run_without_gradme_extras(code):
    """Run code in a fresh interpreter, leaving the test session's imports as is."""
    return subprocess.run(
        [sys.executable, "-c", HIDE_JAX + code],
        capture_output=True,
        text=True,
        check=False,
    )


def test_opt_imports_without_gradme_extras():
    """Without the [gradme] extras, phylo2vec.opt imports and leaves GradME out."""
    result = run_without_gradme_extras(
        "import phylo2vec.opt as opt; print(opt.list_methods(), opt.__all__)"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "['HillClimbing'] ['HillClimbing', 'list_methods']"


def test_gradme_without_extras_says_what_to_install():
    """Asking for GradME without the [gradme] extras names the extras to install."""
    result = run_without_gradme_extras("from phylo2vec.opt import GradME")
    assert result.returncode != 0
    assert "ImportError: GradME needs the optional dependencies" in result.stderr
    assert "pip install phylo2vec[gradme]" in result.stderr


def test_unknown_name_without_gradme_extras():
    """Other missing names still raise Python's usual AttributeError."""
    result = run_without_gradme_extras("import phylo2vec.opt as opt; opt.missing")
    assert result.returncode != 0
    expected = "AttributeError: module 'phylo2vec.opt' has no attribute 'missing'"
    assert expected in result.stderr
