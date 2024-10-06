import pandas as pd
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import time
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from langchain_text_splitters import CharacterTextSplitter


df = pd.read_csv('data/discharge_sample.csv')
discharge_summaries = df.sample(n=5)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt_templates = [
    # Identify the main concept in the text
    (
        "You are a clinician reviewing a patient's discharge summary. \n"
        "Do not include your tasks or instructions.\n"
        "Identify the main diagnosis and/or procedure in the text in one or two words.\n"
        "Examples:\n\n"
        "Text: The patient was admitted for pneumonia.\n"
        "Main concept:: pneumonia \n"
        "Text: The patient underwent hernia repair.\n"
        "Main concept:: hernia repair\n\n"
        "Text: {prompt}\n"
        "Main concept::"
    ),
    # dialogue style template with a system prompt
    (
        "You are a clinician summarizing a patient's discharge summary. "
        "Summarize the following text in one paragraph.\n"
        "Text: {prompt}\n"
        "Summary::"
    ),
]

# Step 1: Identify the main concept in the text from the first 2500 characters
subject_id = 0
concept = ""
main_concepts = []
for index, row in discharge_summaries.iterrows():
        if subject_id != row['subject_id']:
            subject_id = row['subject_id']
            concept = ""
            discharge_note = row['text'][:2500]
            response = generator(
                prompt_templates[0].format(prompt=discharge_note), max_new_tokens=256
            )
            concept = response[0]["generated_text"].split("::")[-1].strip().split("\n")[0]
            print(f"Subject ID: {subject_id}, Main Concept: {concept}")
            main_concepts.append([subject_id, concept, row["text"]])
df = pd.DataFrame(main_concepts, columns=['subject_id', 'concept', 'text'])
df.to_csv('data/main_concepts.csv', index=False)


