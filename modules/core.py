"""Minimal core shim for the server — no UI, no tensorflow."""
import os

import modules.globals


def update_status(message: str, scope: str = 'DLC.CORE') -> None:
    print(f'[{scope}] {message}')
