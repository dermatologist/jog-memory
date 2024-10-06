import pandas as pd
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import time
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from langchain_text_splitters import CharacterTextSplitter


discharge_summaries = pd.read_csv('data/discharge_sample.csv')
main_concepts = pd.read_csv('data/main_concepts.csv')
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt_templates = [
    # Identify the main concept in the text
    (
        "You are a clinician reviewing a patient's discharge summary. "
        "Identify two main concepts in the text separated by ,.\n"
        "Text: {prompt}\n"
        "Main concepts::"
    ),
    # dialogue style template with a system prompt
    # dialogue style template with a system prompt
    (
        "You are a clinician summarizing a patient's discharge summary. "
        "Summarize the following text in one paragraph for the main theme: {concept}\n"
        "Do not include your tasks or instructions.\n"
        "Do not include names, dates, or other identifying information.\n"
        "Text: {prompt}\n"
        "Summary::"
    ),
]

# Step 2: Map documents into a summary < 2500 characters
subject_id = 0
summary = ""
summaries = []
for index, row in discharge_summaries.iterrows():
    excluded = [10000826]
    if row['subject_id'] in excluded:
        continue
    if subject_id != row['subject_id']:
        if len(summary) > 10 and len(summary) < 12000:
            print(f"Subject ID: {subject_id}, \n Summary Length: {len(summary)} \n Summary: {summary}")
            summaries.append([subject_id, summary])
        subject_id = row['subject_id']
        summary = ""
    discharge_note = row['text']
    concept = main_concepts[main_concepts['subject_id'] == subject_id]['concept'].values[0]
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1280, chunk_overlap=5
    )
    split_docs = text_splitter.split_text(discharge_note)
    print(f"Generated {len(split_docs)} documents.")
    print(f"Subject ID: {subject_id}, Concept: {concept}")
    for doc in split_docs:
        response = generator(
            prompt_templates[1].format(concept=concept, prompt=doc), max_new_tokens=128
        )
        full_response = response[0]["generated_text"].split("::")[-1]
        summary += full_response.split("##")[0].strip() + "\n" # remove the system prompt that gets appended
    summaries.append([subject_id, summary])

df = pd.DataFrame(summaries, columns=['subject_id', 'mapped_summary'])
df.to_csv('data/mapped_summaries.csv', index=False)