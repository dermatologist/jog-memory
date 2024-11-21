
import pytest

@staticmethod
def test_find_concept(jm_fixture, capsys):
    text = "The patient was admitted for pneumonia."
    jm_fixture.append_text(text)
    concept = jm_fixture.find_concept()
    assert concept == "pneumonia"



