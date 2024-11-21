
import pytest

@staticmethod
def test_append_text(jm_fixture, capsys):
    text = "The patient was admitted for pneumonia."
    jm_fixture.append_text(text)
    assert jm_fixture.get_text() == text
    jm_fixture.clear_text()
    assert jm_fixture.get_text() == ""