
import pytest


@staticmethod
def test_expand_concept(jm_fixture, capsys):
    concept = "pneumonia"
    assert jm_fixture.expand_concept(concept) != ["pneumonia"]