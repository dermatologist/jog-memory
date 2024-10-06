import pandas as pd
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import time
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from langchain_text_splitters import CharacterTextSplitter
import tqdm

discharge_summaries = pd.read_csv('data/mapped_summaries.csv')
subject_ids = discharge_summaries['subject_id'].unique()
main_concepts = pd.read_csv('data/main_concepts.csv')
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct",
                                             device_map="auto",
                                            #  attn_implementation="flash_attention_2",
                                            #  torch_dtype=torch.float16
                                             )
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
        "Do not include your tasks or instructions.\n"
        "Do not include name, date, or other identifying information.\n"
        "Summarize the following text in one paragraph for the main theme: {concept}\n"
        "Text: {prompt}\n"
        "Summary::"
    ),
]

# https://huggingface.co/docs/transformers/pipeline_tutorial
def stream_data(subject_id):
    docs = []
    for index, notes in discharge_summaries[discharge_summaries['subject_id'] == subject_id].iterrows():
        discharge_note = notes['text']
        concept = main_concepts[main_concepts['subject_id'] == subject_id]['concept'].values[0]
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1280, chunk_overlap=5
        )
        split_docs = text_splitter.split_text(discharge_note)
        for doc in split_docs:
            docs.append(prompt_templates[1].format(concept=concept, prompt=doc))
    for doc in docs:
        yield doc


# Step 2: Map documents into a summary < 2500 characters
summaries = []
excluded = [10000826, 10000032, 10000980, 10001186]
for subject_id in tqdm.tqdm(subject_ids):
    if subject_id in excluded:
        continue
    concept = main_concepts[main_concepts['subject_id'] == subject_id]['concept'].values[0]
    summary = ""
    for response in generator(stream_data(subject_id), max_new_tokens=256):
        full_response = response[0]["generated_text"].split("::")[-1]
        summary += full_response.split("##")[0].strip() + "\n" # remove the system prompt that gets appended
        print(full_response.split("##")[0].strip(), "\n")
        print(f"Subject ID: {subject_id}, Concept: {concept}, Length: {len(summary)}")
        print("---------------------------------------------------------------")

    summaries.append([subject_id, summary, concept])
df = pd.DataFrame(summaries, columns=['subject_id', 'text', 'concept'])
df.to_csv('data/reduced_summaries.csv', index=False)