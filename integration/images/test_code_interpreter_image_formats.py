"""Regression: the code interpreter uploads every base64 image format, not just PNG.

Before this change the `execute_code` tool uploaded only `data:image/png;base64` output; a
JPEG/GIF/WEBP/SVG chart a cell printed stayed inline as a raw data URI in the tool result. The
`<code_interpreter>` tag path accepted `data:image/\\w+;base64`, so the two code-execution paths
disagreed, and `data:image/svg+xml;base64,...` failed on both because `BASE64_IMAGE_URL_PREFIX`
matched `\\w+`, which excludes the `+` in `svg+xml`.

The code interpreter runs on its Jupyter engine, served by `harness.code_interpreter`, whose
kernel runs each cell in a fresh Python process; the tests read what the cell printed from the
stored reply. The cell builds the data URI from the image bytes at run time, so the stored
message (which echoes the cell's source) can be checked for a leftover raw data URI.

Discriminates: passes with the widened prefix; with `BASE64_IMAGE_URL_PREFIX` back to `\\w+` the
SVG cases fail while the PNG cases still pass.
"""

from __future__ import annotations

import json
import re

import pytest

from harness import upstream as reply
from harness.actors import create_user
from harness.chat import ask
from harness.code_interpreter import fake_jupyter

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

CODE_EXECUTION_CONFIG = '/api/v1/configs/code_execution'
TAG_FORMAT = '<code_interpreter type="code" lang="python">'
PATHS = ['tag', 'tool']

# A 1x1 PNG and an empty SVG: the bytes as hex (fed to the cell) and the data URI a failed
# upload would leave behind (asserted absent).
CASES = [
    pytest.param(
        'image/png',
        '89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489'
        '0000000d4944415478da63fccfc0500f000485018084a98c210000000049454e44ae426082',
        'data:image/png;base64,'
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
        id='png',
    ),
    pytest.param(
        'image/svg+xml',
        '3c73766720786d6c6e733d22687474703a2f2f7777772e77332e6f72672f323030302f737667223e3c2f7376673e',
        'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==',
        id='svg',
    ),
]

FILE_URL = re.compile(r'/api/v1/files/[^/]+/content')


def _print_data_uri(mime: str, hex_bytes: str) -> str:
    """A cell that prints a data URI built from `hex_bytes` (so its source holds no base64)."""
    return (
        'import base64\n'
        f"print('data:{mime};base64,' + base64.b64encode(bytes.fromhex('{hex_bytes}')).decode())"
    )


@pytest.fixture(scope='module')
def kernel():
    with fake_jupyter() as jupyter:
        yield jupyter


def _use_kernel(launched, kernel) -> None:
    with launched.client() as client:
        current = client.get(CODE_EXECUTION_CONFIG).json()
        on_kernel = {
            **current,
            'ENABLE_CODE_INTERPRETER': True,
            'CODE_INTERPRETER_ENGINE': 'jupyter',
            'CODE_INTERPRETER_JUPYTER_URL': kernel.base_url,
            'CODE_INTERPRETER_JUPYTER_AUTH': '',
        }
        client.post(CODE_EXECUTION_CONFIG, json=on_kernel).raise_for_status()


@pytest.fixture
def interpreter(instance_with, kernel):
    """An instance of this module's own, on the fake kernel."""
    launched = instance_with({})
    _use_kernel(launched, kernel)
    return launched


def _run(launched, path: str, code: str) -> str:
    """The stored reply after the model ran `code` through the tag or the execute_code tool."""
    if path == 'tag':
        model_reply = reply.text(f'{TAG_FORMAT}\n{code}\n</code_interpreter>')
        function_calling = 'legacy'
    else:
        model_reply = reply.tool_call('execute_code', {'code': code})
        function_calling = 'native'
    launched.upstream.queue(model_reply, reply.text('done'))
    with create_user(launched).client() as client:
        _, message = ask(
            client,
            'run it',
            features={'code_interpreter': True},
            params={'function_calling': function_calling},
        )
    return json.dumps(message)


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('mime, hex_bytes, data_uri', CASES)
def test_a_base64_image_output_is_uploaded_and_served(path, mime, hex_bytes, data_uri, interpreter):
    stored = _run(interpreter, path, _print_data_uri(mime, hex_bytes))
    subtype = mime.split('/')[-1]
    assert FILE_URL.search(stored), (
        f'the {subtype} image a cell printed on the {path} path was not uploaded and served '
        f'from the files API: {stored[-1500:]}'
    )
    assert data_uri not in stored, (
        f'the raw {subtype} data URI survived on the {path} path instead of being replaced by a '
        f'file URL: {stored[-1500:]}'
    )
