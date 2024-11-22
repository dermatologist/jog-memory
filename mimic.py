# Summarize randomly selected notes from the MIMIC-IV dataset.

import pandas as pd
from tqdm import tqdm
import re
from src.jog_memory.jm import JogMemory
from src.jog_memory.rag import JogRag


n_ctx = 2048 + 128
max_tokens = 128 + 128
k=4

df = pd.read_csv('~/data/discharge_5000.csv')
sample = df.sample(n=500) # 30

unique_values = sample['subject_id'].value_counts()
subject_id = unique_values[unique_values == 2].index # <3
subject_id.append(unique_values[unique_values == 3].index)
subject_id = subject_id.unique()

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

# for seach subject_id
for subject_id in tqdm(subject_ids):
    discharge_summaries = df[df['subject_id'] == subject_id]
    jog_memory.clear_text()
    for index, row in tqdm(discharge_summaries.iterrows(), total=discharge_summaries.shape[0]):
        alphaneumeric = re.sub(r'\W+', ' ', row['text']).strip()
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
    print(f"Context: {context}\n")
    print(f"Subject ID: {subject_id}, Main Concept: {concept}, Expanded Concepts: {expanded_concepts}")
    print("Traditional Summary: \n")
    print(jog_memory.summarize(context, concept))
    print("---------------------------------------------------------------\n\n")
    print("Expanded Summary: \n")
    print(jog_memory.summarize(context, concept, expanded_concepts))
    print("---------------------------------------------------------------\n\n")
