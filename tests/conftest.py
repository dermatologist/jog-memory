"""
    Dummy conftest.py for jog_memory.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""

import pytest


@pytest.fixture
def jm_fixture():
    from src.jog_memory.jm import JogMemory
    return JogMemory()

@pytest.fixture
def rag_fixture():
    from src.jog_memory.rag import JogRag
    return JogRag()