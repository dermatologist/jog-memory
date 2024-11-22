
import pytest

@staticmethod
def test_get_summary(jm_fixture, capsys):
    text = """
    Pneumonia is a respiratory condition characterized by inflammation of the air sacs in one or both lungs, which are typically filled with fluid or pus. This infection can be caused by a variety of organisms, including bacteria, viruses, and fungi, leading to a range of symptoms that may include cough with phlegm or pus, fever, chills, and difficulty breathing. The severity of pneumonia can vary greatly, from mild to life-threatening, with the most serious cases occurring in infants, young children, older adults, and individuals with weakened immune systems or chronic health issues.

    The diagnosis of pneumonia involves a review of the patient's medical history, a physical examination, and various laboratory tests. Chest X-rays are commonly used to detect the presence of infection in the lungs, while blood cultures can identify the causative organism. Sputum culture tests help confirm the cause of the infection, and additional tests like urine tests, pulse oximetry, CT scans, and bronchoscopy may be employed to further understand the infection's severity and cause.

    Treatment for pneumonia depends on the type of pathogen causing the infection and the patient's overall health. Antibiotics are used to treat bacterial pneumonia, while antiviral medications are prescribed for viral infections. In cases of fungal pneumonia, antifungal medications are necessary. Alongside these targeted treatments, supportive care such as analgesics to ease pain and cough suppressants to relieve coughing may be provided.

    Prevention of pneumonia is crucial and includes vaccinations, such as the pneumococcal vaccine and the flu vaccine, which can help prevent pneumonia caused by some types of bacteria and the influenza virus. Good hygiene practices, such as regular hand washing and avoiding smoking, can also reduce the risk of developing pneumonia.

    In conclusion, pneumonia is a complex disease with various causes and treatments. It remains a significant health concern globally, especially for high-risk populations. Through prompt diagnosis, appropriate treatment, and preventive measures, the impact of pneumonia can be significantly reduced, improving patient outcomes and reducing the burden on healthcare systems.
    """
    jm_fixture.append_text(text)

    summary = jm_fixture.summarize(concept="pneumonia")
    assert summary is not None
    print(summary)