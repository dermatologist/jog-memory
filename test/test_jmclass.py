import pytest

from jm import JMClass

def test_jmclass():
    assert JMClass().hello() == "Hello, JMClass!"