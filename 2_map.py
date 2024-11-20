import accelerate
import pandas as pd
import tqdm
from langchain_text_splitters import CharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

discharge_summaries = pd.read_csv('~/data/main_concepts.csv')
subject_ids = discharge_summaries['subject_id'].unique()
tokenizer = AutoTokenizer.from_pretrained("KrithikV/MedMobile")
model = AutoModelForCausalLM.from_pretrained("KrithikV/MedMobile",
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
        discharge_note = notes['text']
        concept = discharge_summaries[discharge_summaries['subject_id'] == subject_id]['concept'].values[0]
        expanded_concepts = str(discharge_summaries[discharge_summaries['subject_id'] == subject_id]['expanded_concepts'].values[0])
        print(f"Subject ID: {subject_id}, Concept: {concept}, Expanded Concepts: {expanded_concepts}")
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1280, chunk_overlap=5
        )
        split_docs = text_splitter.split_text(discharge_note)
        for doc in split_docs[:15]: # Limit to 15 chunks because of memory constraints
            if idx == 0:
                docs.append(prompt_templates[0].format(concept=concept, prompt=doc))
            else:
                docs.append(prompt_templates[1].format(concept=concept, prompt=doc, expanded_concepts=expanded_concepts))
    for doc in docs:
        yield doc


# Step 2: Map documents into a summary < 2500 characters
summaries = []
print("Mapping discharge summaries...")
for subject_id in tqdm.tqdm(subject_ids[:10]):  # Limit to 10 summaries
    concept = discharge_summaries[discharge_summaries['subject_id'] == subject_id]['concept'].values[0]
    summary = ["", ""]
    for idx in range(2):
        for response in generator(stream_data(subject_id, idx), max_new_tokens=128):
            full_response = response[0]["generated_text"].split("::")[-1]
            summary[idx] += full_response.split("##")[0].strip() + "\n" # remove the system prompt that gets appended
            print(full_response.split("##")[0].strip(), "\n")
            print(f"Subject ID: {subject_id}, Concept: {concept}, Length: {len(summary[idx])}, Index: {idx}")
            print("---------------------------------------------------------------")

    summaries.append([subject_id, summary[0], summary[1], concept])
df = pd.DataFrame(summaries, columns=['subject_id', 'text', 'jog_memory', 'concept'])
df.to_csv('~/data/mapped_summaries.csv', index=False)