"""Regression: an SVG tool result reached the model as an image input.

`get-file-data` on a wiki that serves SVG natively (`$wgSVGNativeRendering = true`)
returns an MCP image block with `mimeType: image/svg+xml`. `process_tool_result` used
to attach that block as an `input_image` part regardless of its format, and the next
model request carried it as `data:image/svg+xml;base64,…`. DeepSeek accepts only
webp/png/jpeg/gif and rejected the request with HTTP 400
`.messages[15]: You have uploaded an unsupported image…`.

Observed 2026-10-17 on llm.ronzz.org (chat d10e6e62…, request b17c6050…).

`is_model_viewable_image` is the single decision point: only the four formats every
mainstream vision endpoint accepts are viewable; SVG (XML) and the other non-web
formats are not, so they stay display-only files.

An SVG is XML, so it is not dropped from the model's context: `inline_svg_source`
renders its source as a bounded text block in the tool result. Raster images still
travel as `input_image` parts and are never inlined as text.

Discriminates: passes with the guard; fails with `is_model_viewable_image` returning
True for `image/svg+xml` (the previous behaviour).
"""

from __future__ import annotations

import base64

import pytest

pytestmark = pytest.mark.regression


@pytest.fixture(scope="session")
def misc(misc_module):
    return misc_module


@pytest.fixture(scope="session")
def middleware_module(owui_module):
    return owui_module("open_webui.utils.middleware")


@pytest.fixture(scope="session")
def middleware(middleware_module):
    return middleware_module


@pytest.mark.parametrize(
    ("content_type", "expected"),
    [
        ("image/png", True),
        ("image/jpeg", True),
        ("image/webp", True),
        ("image/gif", True),
        ("image/svg+xml", False),
        ("image/bmp", False),
        ("image/tiff", False),
        ("image/heic", False),
        ("image/avif", False),
    ],
)
def test_model_viewable_image_by_content_type(misc, content_type, expected):
    item = {
        'type': 'image',
        'url': '/api/v1/files/x/content',
        'content_type': content_type,
    }

    assert misc.is_model_viewable_image(item) is expected


def test_model_viewable_image_reads_a_data_uri(misc):
    svg = {'type': 'image', 'url': 'data:image/svg+xml;base64,PHN2Zy8+'}
    png = {'type': 'image', 'url': 'data:image/png;base64,iVBORw0KGgo='}

    assert misc.data_uri_media_type(svg['url']) == 'image/svg+xml'
    assert misc.is_model_viewable_image(svg) is False
    assert misc.is_model_viewable_image(png) is True


def test_model_viewable_image_keeps_unknown_types_viewable(misc):
    """Tools that omit a content type keep their previous behaviour."""
    assert misc.is_model_viewable_image({'type': 'image', 'url': '/api/v1/files/x/content'}) is True


@pytest.mark.asyncio
async def test_mcp_image_block_carries_its_mime_type(middleware, monkeypatch):
    stored = {}

    async def fake_get_file_url_from_base64(request, data_uri, metadata, user):
        stored['data_uri'] = data_uri
        return '/api/v1/files/id/content'

    monkeypatch.setattr(middleware, 'get_file_url_from_base64', fake_get_file_url_from_base64)

    _, files, _ = await middleware.process_tool_result(
        request=None,
        tool_function_name='get-file-data',
        tool_result=[{'type': 'image', 'mimeType': 'image/svg+xml', 'data': 'PHN2Zy8+'}],
        tool_type='mcp',
        direct_tool=False,
        metadata={},
        user=None,
    )

    assert stored['data_uri'].startswith('data:image/svg+xml;base64,')
    assert files == [
        {
            'type': 'image',
            'url': '/api/v1/files/id/content',
            'content_type': 'image/svg+xml',
        }
    ]


def _svg_base64(source: str) -> str:
    return base64.b64encode(source.encode('utf-8')).decode('ascii')


@pytest.fixture
def stored_tool_image(middleware, monkeypatch):
    """Replace the storage call so the MCP branch runs without a request/app state."""

    async def fake_get_file_url_from_base64(request, data_uri, metadata, user):
        return '/api/v1/files/id/content'

    monkeypatch.setattr(middleware, 'get_file_url_from_base64', fake_get_file_url_from_base64)


@pytest.mark.asyncio
async def test_mcp_svg_result_reaches_the_model_as_source(middleware, stored_tool_image):
    """An SVG cannot be an image input, but its XML can still reach the model as text."""
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><text>−5</text></svg>'

    text, files, _ = await middleware.process_tool_result(
        request=None,
        tool_function_name='get-file-data',
        tool_result=[{'type': 'image', 'mimeType': 'image/svg+xml', 'data': _svg_base64(svg)}],
        tool_type='mcp',
        direct_tool=False,
        metadata={},
        user=None,
    )

    assert '<text>−5</text>' in text
    assert 'image/svg+xml cannot be sent as a model image input' in text
    assert files[0]['content_type'] == 'image/svg+xml'


@pytest.mark.asyncio
async def test_mcp_raster_result_does_not_inline_source(middleware, stored_tool_image):
    text, files, _ = await middleware.process_tool_result(
        request=None,
        tool_function_name='get-file-data',
        tool_result=[{'type': 'image', 'mimeType': 'image/png', 'data': _svg_base64('png-bytes')}],
        tool_type='mcp',
        direct_tool=False,
        metadata={},
        user=None,
    )

    assert '<svg' not in text
    assert files[0]['content_type'] == 'image/png'


@pytest.mark.asyncio
async def test_large_svg_source_is_truncated(middleware, stored_tool_image):
    svg = '<svg>' + ('x' * (middleware.DEFAULT_INLINE_SVG_SOURCE_CHARS + 5000)) + '</svg>'

    text, _, _ = await middleware.process_tool_result(
        request=None,
        tool_function_name='get-file-data',
        tool_result=[{'type': 'image', 'mimeType': 'image/svg+xml', 'data': _svg_base64(svg)}],
        tool_type='mcp',
        direct_tool=False,
        metadata={},
        user=None,
    )

    assert 'truncated:' in text
    assert 'characters omitted' in text
    assert text.count('x') <= middleware.DEFAULT_INLINE_SVG_SOURCE_CHARS


@pytest.mark.asyncio
async def test_svg_data_uri_result_reaches_the_model_as_source(middleware):
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><text>π</text></svg>'
    data_uri = 'data:image/svg+xml;base64,' + _svg_base64(svg)

    text, files, _ = await middleware.process_tool_result(
        request=None,
        tool_function_name='make_graph',
        tool_result=data_uri,
        tool_type='action',
        direct_tool=False,
        metadata={},
        user=None,
    )

    assert '<text>π</text>' in text
    assert files == [{'type': 'image', 'url': data_uri}]


@pytest.mark.parametrize(
    ('context_window', 'used_tokens', 'expected'),
    [
        # 30% of free context when usage is known.
        (32_000, 8_000, 21_600),
        # 15% of the whole window when usage is unknown.
        (32_000, None, 14_400),
        # Ceiling clamps a 1M-token window.
        (1_000_000, 0, 200_000),
        # Floor clamps a nearly full context.
        (20_000, 19_000, 10_000),
        # No context window information.
        (None, 1_000, 20_000),
    ],
)
def test_inline_svg_budget_follows_the_context_window(middleware, context_window, used_tokens, expected):
    models = {'base': {'context_window': context_window}} if context_window else {}
    model = {'info': {'base_model_id': 'base'}} if context_window else {}
    usage = {'prompt_tokens': used_tokens} if used_tokens is not None else None

    assert middleware.resolve_inline_svg_char_budget(models, model, usage) == expected


def test_inline_svg_budget_reads_the_model_own_window(middleware):
    models = {'direct': {'context_window': 64_000}}
    model = models['direct']

    assert middleware.resolve_inline_svg_char_budget(models, model, {'prompt_tokens': 16_000}) == 43_200


@pytest.mark.asyncio
async def test_svg_budget_is_shared_across_images_in_one_result(middleware, stored_tool_image):
    """Two SVGs in one result split the per-result budget instead of each claiming it."""
    svg = '<svg>' + ('x' * 15_000) + '</svg>'
    item = {'type': 'image', 'mimeType': 'image/svg+xml', 'data': _svg_base64(svg)}

    text, _, _ = await middleware.process_tool_result(
        request=None,
        tool_function_name='get-file-data',
        tool_result=[item, dict(item)],
        tool_type='mcp',
        direct_tool=False,
        metadata={},
        user=None,
        inline_svg_char_budget=20_000,
    )

    assert text.count('x') <= 20_000
