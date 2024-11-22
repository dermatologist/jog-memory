# Summarize randomly selected notes from the MIMIC-IV dataset.

import pandas as pd
from tqdm import tqdm
import re
from src.jog_memory.jm import JogMemory
from src.jog_memory.rag import JogRag


n_ctx = 2048
max_tokens = 256


df = pd.read_csv('~/data/discharge_5000.csv')
sample = df.sample(n=500) # 30
subject_id = sample['subject_id'].unique()
subject_id = subject_id[0:30]
print(f"Subject IDs: {subject_id}")
discharge_summaries = df[df['subject_id'].isin(subject_id)]

jog_memory = JogMemory(
    n_ctx=n_ctx,
    max_tokens=max_tokens,
)
jog_rag = JogRag()

subject_id = 0
for index, row in tqdm(discharge_summaries.iterrows(), total=discharge_summaries.shape[0]):
    alphaneumeric = re.sub(r'\W+', ' ', row['text']).strip()
    if subject_id != row['subject_id'] and subject_id != 0:
        jog_memory.append_text(alphaneumeric)
    else:
        # Main prodessing
        # print(jog_memory.get_text())
        print(index, subject_id)

        # identify main concept and expanded concepts
        concept = jog_memory.find_concept()
        expanded_concepts = jog_memory.expand_concept(concept)

        # continue if concept is not found or expanded concepts list is empty
        if concept == "" or len(expanded_concepts) == 0:
            jog_memory.clear_text()
            jog_memory.append_text(alphaneumeric)
            subject_id = row['subject_id']
            continue

        # RAG if length of text exceeds context window size
        if len(jog_memory.get_text()) > (n_ctx-300):
            docs = jog_rag.split_text(jog_memory.get_text(), subject_id, concept, expanded_concepts)
            context = jog_rag.get_context(concept, expanded_concepts)
        else:
            context = jog_memory.get_text()

        # Summarize the context
        print(f"Subject ID: {subject_id}, Main Concept: {concept}, Expanded Concepts: {expanded_concepts}")
        print("Traditional Summary: \n")
        print(jog_memory.summarize(concept))
        print("---------------------------------------------------------------\n\n")
        print("Expanded Summary: \n")
        print(jog_memory.summarize(concept, expanded_concepts))
        print("---------------------------------------------------------------\n\n")

        # Clear the text
        jog_memory.clear_text()
        jog_memory.append_text(row['text'])
        subject_id = row['subject_id']
