import accelerate
import pandas as pd
import tqdm
from langchain_text_splitters import CharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import re
import random
import os

discharge_summaries = pd.read_csv('~/data/map2_summaries.csv')
main_concepts = pd.read_csv('~/data/main_concepts.csv')
subject_ids = discharge_summaries['subject_id'].unique()
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct",
                                             device_map="auto",
                                            #  attn_implementation="flash_attention_2",
                                            #  torch_dtype=torch.float16
                                             )
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt_templates = [
    (
        "You are a clinician summarizing a patient's discharge summary. "
        "Do not include your tasks or instructions.\n"
        "Do not include name, date, or other identifying information.\n"
        "Summarize the following text in one paragraph for the main theme: {concept}\n"
        "Text: {prompt}\n"
        "Summary::"
    ),
    (
        "You are a clinician summarizing a patient's discharge summary. "
        "Do not include your tasks or instructions.\n"
        "Do not include name, date, or other identifying information.\n"
        "Summarize the following text in one paragraph for the main theme: {concept}\n"
        "Mention if any of the following are present: {expanded_concepts}\n"
        "Text: {prompt}\n"
        "Summary::"
    ),
]

def trim_after_last_period(sentence):
    # Find the last occurrence of a period
    last_period_index = sentence.rfind('.')

    # If a period is found, trim the sentence after the last period
    if last_period_index != -1:
        return re.sub('\d', '*',sentence[:last_period_index + 1])
    else:
        # If no period is found, return the original sentence
        return re.sub('\d', '*',sentence)


# https://huggingface.co/docs/transformers/pipeline_tutorial
def stream_data(subject_id, idx=1):
    docs = []
    for index, notes in discharge_summaries[discharge_summaries['subject_id'] == subject_id].iterrows():
        if idx == 0:
            discharge_note = notes['text']
        else:
            discharge_note = notes['jog_memory']
        concept = discharge_summaries[discharge_summaries['subject_id'] == subject_id]['concept'].values[0]
        expanded_concepts = str(main_concepts[main_concepts['subject_id'] == subject_id]['expanded_concepts'].values[0])
        print(f"Subject ID: {subject_id}, Concept: {concept}, Expanded Concepts: {expanded_concepts}")
        if idx == 0:
            docs.append(prompt_templates[0].format(concept=concept, prompt=discharge_note))
        else:
            docs.append(prompt_templates[1].format(concept=concept, prompt=discharge_note, expanded_concepts=expanded_concepts))
    for doc in docs:
        yield doc

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

# Step 2: Map documents into a summary < 2500 characters
summaries = []
print("Reducing discharge summaries...")
for subject_id in tqdm.tqdm(subject_ids):
    concept = discharge_summaries[discharge_summaries['subject_id'] == subject_id]['concept'].values[0]
    expanded_concepts = str(main_concepts[main_concepts['subject_id'] == subject_id]['expanded_concepts'].values[0])
    summary = ["", ""]
    for idx in range(2):
        for response in generator(stream_data(subject_id, idx), max_new_tokens=200):
            full_response = response[0]["generated_text"].split("::")[-1]
            summary[idx] += full_response.split("##")[0].strip() + "\n" # remove the system prompt that gets appended
            print(full_response.split("##")[0].strip(), "\n")
            print(f"Subject ID: {subject_id}, Concept: {concept}, Length: {len(summary[idx])}, Index: {idx}")
            print("---------------------------------------------------------------")

    summaries.append([subject_id, summary[0], summary[1], concept])
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