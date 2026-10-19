"""Regression: execute_code stages the files listed in files_to_save.

The code interpreter has its own filesystem. A file written by a cell is invisible to the
chat and to other tools until the model lists its path in ``files_to_save``. The server
should then copy it into the chat, attach it to the message, and return a server path in
``files[]`` that a file-upload tool can consume.

The Jupyter harness runs each cell in a fresh process on this machine, so the kernel and
the backend share the upload directory. That is the branch this test exercises.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

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
UPLOAD_DIR = Path(os.environ.get('OPEN_WEBUI_UPLOAD_DIR', '/tmp/open-webui-uploads'))


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
    launched = instance_with({})
    _use_kernel(launched, kernel)
    return launched


def _execute_code_result(message: dict) -> dict:
    for item in message.get('output') or []:
        if item.get('type') != 'function_call_output':
            continue
        for part in item.get('output') or []:
            text = part.get('text')
            if part.get('type') != 'input_text' or not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and 'files' in parsed:
                return parsed
    return {}


def test_files_to_save_returns_a_server_path_and_attaches_the_file(interpreter):
    name = f'staged-{uuid.uuid4().hex[:8]}.svg'
    path = UPLOAD_DIR / name
    code = (
        'from pathlib import Path\n'
        f'path = Path(UPLOAD_DIR) / "{name}"\n'
        'path.write_text(\'<svg xmlns="http://www.w3.org/2000/svg"></svg>\')\n'
        'print(path)\n'
    )

    interpreter.upstream.queue(
        reply.tool_call('execute_code', {'code': code, 'files_to_save': [str(path)]}),
        reply.text('done'),
    )

    try:
        with create_user(interpreter).client() as client:
            _, message = ask(
                client,
                'make the file',
                features={'code_interpreter': True},
                params={'function_calling': 'native'},
            )

        result = _execute_code_result(message)
        files = result.get('files') or []
        assert files, f'the staged file is missing from the tool result: {json.dumps(message)[-2000:]}'

        entry = files[0]
        assert entry.get('name') == name
        assert str(entry.get('url', '')).startswith('/api/v1/files/')
        assert entry.get('path') and entry['path'] != str(path), (
            'files[].path must be the server path, not the interpreter path: '
            f'{entry}'
        )

        attached = [file for file in (message.get('files') or []) if file.get('name') == name]
        assert attached, f'the staged file was not attached to the message: {message.get("files")}'
        assert attached[0].get('id') == entry.get('id')
    finally:
        path.unlink(missing_ok=True)
