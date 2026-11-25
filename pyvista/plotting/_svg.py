"""Repair multiline text in SVG output produced by GL2PS.

``vtkGL2PSExporter`` writes every text actor as a single ``<text>`` element
whose string still contains the raw newline characters. Newlines inside SVG
text collapse to whitespace, so all lines render on top of each other. GL2PS
also writes the string without escaping XML special characters. The helpers
here split each text element into one absolutely positioned ``<tspan>`` per
line and escape the text content.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

_TEXT_RE = re.compile(
    r'<text\b[^>]*/>'
    r'|<text\b(?P<attrs>[^>]*)>(?P<content>.*?)</text>',
    re.DOTALL,
)
_ATTR_RE = re.compile(r'(?P<name>[\w:-]+)="(?P<value>[^"]*)"')

# Presentation attributes copied onto every line so each <tspan> carries its
# own size, family and color for tools that do not resolve inheritance.
_PRESENTATION_ATTRS = (
    'fill',
    'fill-opacity',
    'font-family',
    'font-size',
    'font-style',
    'font-weight',
    'text-anchor',
)


def _escape_xml_text(text: str) -> str:
    """Escape the XML special characters in a text node."""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _format_number(value: float) -> str:
    """Format a coordinate without a trailing ``.0``."""
    rounded = round(value, 3)
    if float(rounded).is_integer():
        return str(int(rounded))
    return f'{rounded:.3f}'.rstrip('0').rstrip('.')


def _as_float(attrs: dict[str, str], name: str, default: float) -> float:
    """Read a numeric attribute, falling back when it is missing or invalid."""
    try:
        return float(attrs[name])
    except (KeyError, TypeError, ValueError):
        return default


def _tspan_attributes(attrs: dict[str, str]) -> str:
    """Return the presentation attributes shared by every line."""
    return ''.join(f' {name}="{attrs[name]}"' for name in _PRESENTATION_ATTRS if name in attrs)


def _split_text_element(match: re.Match[str]) -> str:
    """Replace one GL2PS ``<text>`` element with positioned line tspans."""
    if match.group('attrs') is None:
        # Self-closing element such as ``<text ... />``; nothing to split.
        return match.group(0)
    attrs_text = match.group('attrs')
    content = match.group('content')
    if '<tspan' in content:
        return match.group(0)
    attrs = dict(_ATTR_RE.findall(attrs_text))

    if '\r' in content:
        content = content.replace('\r\n', '\n').replace('\r', '\n')
    lines = content.split('\n')

    if len(lines) == 1:
        # GL2PS does not escape special characters even for single-line text.
        return f'<text{attrs_text}>{_escape_xml_text(content)}</text>'

    font_size = _as_float(attrs, 'font-size', 16.0)
    x = attrs.get('x', '0')
    y = _as_float(attrs, 'y', 0.0)
    root_dy = _as_float(attrs, 'dy', 0.0)

    # GL2PS chooses ``dy`` from the vertical justification: ``0`` anchors the
    # last line, ``font-size / 2`` the middle line and ``font-size`` the first
    # line. ``y + dy`` is therefore the baseline of line ``anchor``.
    if root_dy >= 0.75 * font_size:
        anchor = 0.0
    elif root_dy <= 0.25 * font_size:
        anchor = float(len(lines) - 1)
    else:
        anchor = (len(lines) - 1) / 2.0

    first_baseline = y + root_dy - anchor * font_size
    presentation = _tspan_attributes(attrs)

    tspans = []
    for index, line in enumerate(lines):
        baseline = first_baseline + index * font_size
        tspans.append(
            f'<tspan x="{x}" y="{_format_number(baseline)}" dy="0"'
            f'{presentation}>{_escape_xml_text(line)}</tspan>'
        )
    return f'<text{attrs_text}>{"".join(tspans)}</text>'


def split_multiline_text(svg: str) -> str:
    """Split newline-separated SVG text into visible ``<tspan>`` lines.

    Parameters
    ----------
    svg : str
        SVG document emitted by ``vtkGL2PSExporter``.

    Returns
    -------
    str
        The SVG document with every ``<text>`` element XML-escaped and its
        explicit newlines represented by separate, positioned ``<tspan>``
        elements. If the result is not well-formed XML, ``svg`` is returned
        unchanged so a caller never receives a corrupted document.

    """
    try:
        result = _TEXT_RE.sub(_split_text_element, svg)
        ET.fromstring(result.encode('utf-8'))
    except (ET.ParseError, ValueError):
        return svg
    return result
