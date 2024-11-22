# Summarize randomly selected notes from the MIMIC-IV dataset.

import pandas as pd
from tqdm import tqdm
import re
import random
import os
from src.jog_memory.jm import JogMemory
from src.jog_memory.rag import JogRag


n_ctx = 2048 + 256
max_tokens = 128 + 128
k=5

df = pd.read_csv('~/data/discharge_5000.csv')
sample = df.sample(n=500) # 30

subject_id = sample['subject_id'].unique()

# subject_id = sample['subject_id'].unique()
subject_ids = subject_id[0:30]
print(f"Subject IDs: {subject_id}")
# discharge_summaries = df[df['subject_id'].isin(subject_id)]

jog_memory = JogMemory(
    n_ctx=n_ctx,
    max_tokens=max_tokens,
)
jog_rag = JogRag(
    n_ctx=n_ctx,
)

# Anominize and save the summaries
def anonymize(summary, concept):
    first = random.choice([0,1])
    second = 1 - first
    content = f"""
        Concept: {concept}

        Summary A:
        {summary[first]}

        Summary B:
        {summary[second]}

        ---------------------------------------------------------------
        """
    return content

count = 0
# for seach subject_id
for subject_id in tqdm(subject_ids):
    discharge_summaries = df[df['subject_id'] == subject_id]
    # if rows exceed 5, continue
    if discharge_summaries.shape[0] > 3:
        continue
    jog_memory.clear_text()
    for index, row in tqdm(discharge_summaries.iterrows(), total=discharge_summaries.shape[0]):
        alphaneumeric = re.sub(r'\W+', ' ', row['text']).strip()
        alphaneumeric = alphaneumeric.replace("_", "")
        jog_memory.append_text(alphaneumeric)
    # Main prodessing
    # print(jog_memory.get_text())
    print(f"Length of text: {len(jog_memory.get_text())}")
    print(index, subject_id)

    # identify main concept and expanded concepts
    concept = jog_memory.find_concept()
    expanded_concepts = jog_memory.expand_concept(concept)

    # continue if concept is not found or expanded concepts list is empty
    if concept == "" or len(expanded_concepts) == 0:
        continue


    # RAG if length of text exceeds context window size
    if len(jog_memory.get_text()) > (n_ctx-300):
        docs = jog_rag.split_text(jog_memory.get_text(), subject_id, concept, expanded_concepts, k=k)
        context = jog_rag.get_context(concept, expanded_concepts, k=k)
    else:
        context = jog_memory.get_text()

    # Summarize the context
    traditional = jog_memory.summarize(context, concept)
    expanded = jog_memory.summarize(context, concept, expanded_concepts)
    print(f"Context: {context}\n")
    print(f"Subject ID: {subject_id}, Main Concept: {concept}, Expanded Concepts: {expanded_concepts}")
    print("Traditional Summary: \n")
    print(traditional)
    print("---------------------------------------------------------------\n\n")
    print("Expanded Summary: \n")
    print(expanded)
    print("---------------------------------------------------------------\n\n")

    # Save the summaries
    content = f"""
    Subject ID: {subject_id} | Concept: {concept} | Expanded Concepts: {expanded_concepts}

    Default Summary:
    {traditional}

    JOG Memory Summary:
    {expanded}

    ---------------------------------------------------------------
    """
    with open(os.environ['HOME'] + '/data/report.txt', 'a') as f:
        f.write(content)

    with open(os.environ['HOME'] + '/data/report_anon.txt', 'a') as f:
        f.write(anonymize([traditional, expanded], concept))

    if count < 10:
        count += 1
    else:
        exit()