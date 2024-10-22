import tqdm
import os
import re
import random
import time
import pandas as pd
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = GoogleGenerativeAI(model="models/gemini-1.5-flash-latest")

discharge_summaries = pd.read_csv('~/data/main_concepts.csv')
subject_ids = discharge_summaries['subject_id'].unique()
prompt_templates = [
    (
        "You are a clinician summarizing a patient's discharge summary. "
        "Do not include your tasks or instructions.\n"
        "Do not include name, date, or other identifying information.\n"
        "Context: {prompt}\n"
        "Summarize the above context in one paragraph for the theme: {concept} \n"
        "Summary::"
    ),
    (
        "You are a clinician summarizing a patient's discharge summary. "
        "Do not include your tasks or instructions.\n"
        "Do not include name, date, or other identifying information.\n"
        "Context: {prompt}\n"
        "Summarize the above context in one paragraph for the themes: {expanded_concepts}, {concept} \n"
        "Summary::"
    ),
]

prompt = ["", ""]
prompt[0] = PromptTemplate(prompt_templates[0])
prompt[1] = PromptTemplate(prompt_templates[1])



# Anominize and save the summaries
def anonymize(summary, concept):
    first = random.choice([0,1])
    second = 1 - first
    content = f"""
        Concept: {concept}

        Summary A:
        {trim_after_last_period(summary[first])}

        Summary B:
        {trim_after_last_period(summary[second])}

        ---------------------------------------------------------------
        """
    return content

def collate_notes(subject_id):
    discharge_note = ""
    for index, notes in discharge_summaries[discharge_summaries['subject_id'] == subject_id].iterrows():
        discharge_note += notes['text']
        discharge_note += " \n"
    return discharge_note

def trim_after_last_period(sentence):
    # Find the last occurrence of a period
    last_period_index = sentence.rfind('.')

    # If a period is found, trim the sentence after the last period
    if last_period_index != -1:
        return re.sub('\d', '*',sentence[:last_period_index + 1])
    else:
        # If no period is found, return the original sentence
        return re.sub('\d', '*',sentence)

summaries = []
print("Working ...")
for subject_id in tqdm.tqdm(subject_ids):
    concept = discharge_summaries[discharge_summaries['subject_id'] == subject_id]['concept'].values[0]
    expanded_concepts = str(discharge_summaries[discharge_summaries['subject_id'] == subject_id]['expanded_concepts'].values[0])
    _input = {
        "prompt": collate_notes(subject_id),
        "concept": concept,
        "expanded_concepts": expanded_concepts
    }
    chain_0 = prompt[0] | llm | StrOutputParser()
    chain_1 = prompt[1] | llm | StrOutputParser()
    summary = [chain_0.invoke(_input), chain_1.invoke(_input)]
    time.sleep(15)
    summaries.append([subject_id, summary[0], summary[1], concept, expanded_concepts])
    content = f"""
    Subject ID: {subject_id} | Concept: {concept} | Expanded Concepts: {expanded_concepts}

    Default Summary:
    {trim_after_last_period(summary[0])}

    JOG Memory Summary:
    {trim_after_last_period(summary[1])}

    ---------------------------------------------------------------
    """
    with open(os.environ['HOME'] + '/data/report.txt', 'a') as f:
        f.write(content)

    with open(os.environ['HOME'] + '/data/report_anon.txt', 'a') as f:
        f.write(anonymize(summary, concept))

df = pd.DataFrame(summaries, columns=['subject_id', 'text', 'jog_memory', 'concept'])
df.to_csv('~/data/final_summaries.csv', index=False)