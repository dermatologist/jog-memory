import pandas as pd
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_text_splitters import CharacterTextSplitter
import tqdm
import accelerate

discharge_summaries = pd.read_csv('data/mapped_summaries.csv')
main_concepts = pd.read_csv('data/main_concepts.csv')
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
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1280, chunk_overlap=5
        )
        split_docs = text_splitter.split_text(discharge_note)
        for doc in split_docs:
            if idx == 0:
                docs.append(prompt_templates[0].format(concept=concept, prompt=doc))
            else:
                docs.append(prompt_templates[1].format(concept=concept, prompt=doc, expanded_concepts=expanded_concepts))
    for doc in docs:
        yield doc


# Step 2: Map documents into a summary < 2500 characters
summaries = []
excluded = [10000826, 10000032, 10000980, 10001186]
print("Mapping discharge summaries...(2)")
for subject_id in tqdm.tqdm(subject_ids):
    if subject_id in excluded:
        continue
    concept = discharge_summaries[discharge_summaries['subject_id'] == subject_id]['concept'].values[0]
    summary = ["", ""]
    for idx in range(2):
        if idx == 0 and len (discharge_summaries[discharge_summaries['subject_id'] == subject_id]['text'].values[0]) < 2000:
            summary[idx] = discharge_summaries[discharge_summaries['subject_id'] == subject_id]['text'].values[0]
            continue
        if idx == 1 and len (discharge_summaries[discharge_summaries['subject_id'] == subject_id]['jog_memory'].values[0]) < 2000:
            summary[idx] = discharge_summaries[discharge_summaries['subject_id'] == subject_id]['jog_memory'].values[0]
            continue
        for response in generator(stream_data(subject_id, idx), max_new_tokens=128):
            full_response = response[0]["generated_text"].split("::")[-1]
            summary[idx] += full_response.split("##")[0].strip() + "\n" # remove the system prompt that gets appended
            print(full_response.split("##")[0].strip(), "\n")
            print(f"Subject ID: {subject_id}, Concept: {concept}, Length: {len(summary[idx])}, Index: {idx}")
            print("---------------------------------------------------------------")

    summaries.append([subject_id, summary[0], summary[1], concept])
df = pd.DataFrame(summaries, columns=['subject_id', 'text', 'jog_memory', 'concept'])
df.to_csv('data/map2_summaries.csv', index=False)