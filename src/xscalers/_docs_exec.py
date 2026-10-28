"""Custom Python fence for the docs: notebook-style last-expression output.

Extends markdown-exec so a bare expression statement at the end of a code
block renders as the object's rich HTML repr (xarray's ``_repr_html_``),
falling back to a plain text repr otherwise. The displayed source is left
untouched, so guides read like ordinary Python.

The fence runs with HTML output forced: reprs are HTML, and plain text
must be authored as ``<p>...</p>`` blocks (Zensical strips newlines from
markdown output anyway, so HTML paragraphs are more reliable).
"""

from __future__ import annotations

import ast
import textwrap
from typing import TYPE_CHECKING, Any

from markdown_exec import base_format, validator  # noqa: F401  (re-exported)
from markdown_exec._internal.formatters.python import _run_python

if TYPE_CHECKING:
    from markdown.core import Markdown


def _repr_transform(source: str) -> tuple[str, str]:
    """Wrap a bare trailing expression into a rich-repr print.

    Returns the source to run (last expression captured and printed as
    HTML repr) and the source to display (unchanged).
    """
    tree = ast.parse(source)
    body = tree.body
    if not body or not isinstance(body[-1], ast.Expr):
        return source, source
    last = body[-1]
    # calls (print, asserts) are statements whose value is not a result
    if isinstance(last.value, ast.Constant | ast.Call):
        return source, source
    if last.lineno is None or last.end_lineno is None:
        return source, source
    lines = source.splitlines(keepends=True)
    first = last.lineno - 1
    end = last.end_lineno - 1
    prefix = lines[first][: last.col_offset]
    suffix = lines[end][last.end_col_offset :]
    segment = ast.get_source_segment(source, last)
    if segment is None:
        return source, source
    indented = textwrap.indent(segment.rstrip(), "    ")
    run_source = (
        f"{prefix}_docs_val = (\n{indented}\n)\n"
        "print(\n"
        "    _docs_val._repr_html_()\n"
        "    if hasattr(_docs_val, '_repr_html_')\n"
        "    else '<pre><code>'"
        " + __import__('html').escape(f'{_docs_val!r}') + '</code></pre>'\n"
        ")\n"
    )
    input_source = "".join(lines[:first]) + run_source + suffix
    input_source += "".join(lines[end + 1 :])
    return input_source, source


def formatter(
    source: str,
    language: str,
    css_class: str,
    options: dict[str, Any],
    md: Markdown,
    **kwargs: Any,
) -> str:
    """Execute the fence and return HTML, forcing HTML output mode."""
    opts = dict(options)
    opts["html"] = True
    return str(
        base_format(
            language=language,
            run=_run_python,
            code=source,
            md=md,
            transform_source=_repr_transform,
            **opts,
        )
    )
