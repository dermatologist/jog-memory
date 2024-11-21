
import pytest

@staticmethod
def test_find_concept(jm_fixture, capsys):
    text = "The patient was admitted for pneumonia."
    jm_fixture.append_text(text)
    concept = jm_fixture.find_concept()
    assert concept == "pneumonia"

def test_expand_concept(jm_fixture, capsys):
    concept = "pneumonia"
    expanded_concepts = jm_fixture.expand_concept(concept)
    print(f"Expanded Concepts: {expanded_concepts}")
    assert expanded_concepts == ['pulmonary infection', 'severe pneumonia', 'lung infection', 'necrotizing pneumonia', 'viral pneumonia', 'bronchopneumonia', 'bronchitis', 'respiratory infection', 'aspiration pneumonia', 'bacterial pneumonia']



