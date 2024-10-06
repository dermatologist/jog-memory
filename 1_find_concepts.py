import pandas as pd
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import time
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from langchain_text_splitters import CharacterTextSplitter


discharge_summaries = pd.read_csv('data/discharge_sample.csv')
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt_templates = [
    # Identify the main concept in the text
    (
        "You are a clinician reviewing a patient's discharge summary. "
        "Identify the main diagnostic or procedure concept in the text in one or two words.\n"
        "Example: pneumonia, hernia repair\n"
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
            main_concepts.append([subject_id, concept])
df = pd.DataFrame(main_concepts, columns=['subject_id', 'concept'])
df.to_csv('data/main_concepts.csv', index=False)


