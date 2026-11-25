"""Tests for multiline text in SVG graphic export."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

import pyvista as pv
from pyvista.plotting._svg import split_multiline_text

_SVG_NS = '{http://www.w3.org/2000/svg}'


def test_split_multiline_text_creates_a_tspan_per_line():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<text x="10" y="100" font-size="20" dy="0">A\nB\nC</text>'
        '</svg>'
    )
    out = split_multiline_text(svg)
    root = ET.fromstring(out)
    tspans = root.findall(f'.//{_SVG_NS}tspan')
    assert [t.text for t in tspans] == ['A', 'B', 'C']
    ys = [float(t.get('y')) for t in tspans]
    assert ys[1] > ys[0]
    assert ys[2] > ys[1]
    assert ys[1] - ys[0] == pytest.approx(20)
    assert ys[2] - ys[1] == pytest.approx(20)
    # Bottom anchored: the original baseline is the last line.
    assert ys[2] == pytest.approx(100)


def test_split_multiline_text_keeps_empty_lines():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<text x="0" y="100" font-size="10" dy="10">A\n\nC</text>'
        '</svg>'
    )
    root = ET.fromstring(split_multiline_text(svg))
    tspans = root.findall(f'.//{_SVG_NS}tspan')
    assert [t.text or '' for t in tspans] == ['A', '', 'C']
    ys = [float(t.get('y')) for t in tspans]
    assert ys[2] - ys[0] == pytest.approx(20)


def test_split_multiline_text_line_pitch_is_font_size():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<text x="0" y="200" font-size="40" dy="0">A\nB\nC</text>'
        '</svg>'
    )
    root = ET.fromstring(split_multiline_text(svg))
    ys = [float(t.get('y')) for t in root.findall(f'.//{_SVG_NS}tspan')]
    assert ys == [120.0, 160.0, 200.0]  # bottom anchored: last line on y


@pytest.mark.parametrize(
    ('dy', 'expected_first_y'),
    [('40', 0.0), ('20', -60.0), ('0', -120.0)],  # top, center (3 lines), bottom
)
def test_split_multiline_text_anchor_endpoints(dy, expected_first_y):
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        f'<text x="0" y="-40" font-size="40" dy="{dy}">A\nB\nC</text>'
        '</svg>'
    )
    root = ET.fromstring(split_multiline_text(svg))
    ys = [float(t.get('y')) for t in root.findall(f'.//{_SVG_NS}tspan')]
    assert ys[0] == pytest.approx(expected_first_y)
    assert ys[1] - ys[0] == pytest.approx(40)


def test_split_multiline_text_normalizes_crlf():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<text x="0" y="80" font-size="20" dy="0">X\r\nY\rZ</text>'
        '</svg>'
    )
    root = ET.fromstring(split_multiline_text(svg))
    assert [t.text for t in root.findall(f'.//{_SVG_NS}tspan')] == ['X', 'Y', 'Z']


def test_split_multiline_text_ignores_self_closing_text():
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><text x="0" y="0" /></svg>'
    assert split_multiline_text(svg) == svg


@pytest.mark.parametrize(
    ('raw', 'escaped'),
    [
        ('<A&B>', '&lt;A&amp;B&gt;'),
        ('café 日本語', 'café 日本語'),
        ('"quotes" \'apos\'', '"quotes" \'apos\''),
    ],
)
def test_split_multiline_text_escapes_content(raw, escaped):
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        f'<text x="0" y="0" font-size="10" dy="0">{raw}</text>'
        '</svg>'
    )
    out = split_multiline_text(svg)
    assert escaped in out
    root = ET.fromstring(out)  # must remain valid XML
    text = root.find(f'.//{_SVG_NS}text')
    assert ''.join(text.itertext()) == raw


def test_single_line_text_is_unchanged_apart_from_escaping():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<text x="1" y="2" font-size="9" dy="0">Hello</text>'
        '</svg>'
    )
    out = split_multiline_text(svg)
    assert '<text' in out
    assert '<tspan' not in out
    assert '>Hello</text>' in out


def test_save_graphic_svg_multiline_text(tmp_path):
    svg_path = tmp_path / 'scene.svg'
    pl = pv.Plotter(window_size=(800, 600), off_screen=True)
    pl.add_mesh(pv.Sphere(), color='white')
    pl.add_title('First Line\nSecond Line', font_size=18, color='red')
    pl.add_text('Alpha\nBeta', position=(100, 200), font_size=14, color='blue')
    pl.save_graphic(svg_path)
    pl.close()

    svg = svg_path.read_text(encoding='utf-8')
    root = ET.fromstring(svg)  # special characters must not break parsing
    tspans = root.findall(f'.//{_SVG_NS}tspan')
    contents = [t.text or '' for t in tspans]
    # add_title prepends a blank line; the two requested lines are present.
    assert 'First Line' in contents
    assert 'Second Line' in contents
    assert 'Alpha' in contents
    assert 'Beta' in contents

    # Every visible line has its own vertical coordinate, strictly increasing
    # within each text element.
    for text in root.iter(f'{_SVG_NS}text'):
        ys = [float(t.get('y')) for t in text.iter(f'{_SVG_NS}tspan')]
        assert ys == sorted(ys)
        if len(ys) > 1:
            assert np.all(np.diff(ys) > 0)


def test_save_graphic_svg_special_characters_round_trip(tmp_path):
    svg_path = tmp_path / 'special.svg'
    pl = pv.Plotter(window_size=(400, 300), off_screen=True)
    pl.add_text('<A&B>\nCafé 中文', position=(20, 40), font_size=12)
    pl.save_graphic(svg_path)
    pl.close()

    root = ET.parse(svg_path).getroot()
    all_text = [''.join(t.itertext()) for t in root.iter(f'{_SVG_NS}text')]
    joined = '\n'.join(all_text)
    assert '<A&B>' in joined
    assert 'Café 中文' in joined
