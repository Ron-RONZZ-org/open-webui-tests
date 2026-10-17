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

Discriminates: passes with the guard; fails with `is_model_viewable_image` returning
True for `image/svg+xml` (the previous behaviour).
"""

from __future__ import annotations

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
