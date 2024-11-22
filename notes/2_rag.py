from langchain_huggingface import HuggingFaceEmbeddings
import accelerate
import pandas as pd
import tqdm
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import shutil
import re

import logging
logging.getLogger("langchain_text_splitters.base").setLevel(logging.ERROR)

discharge_summaries = pd.read_csv('~/data/main_concepts.csv')
subject_ids = discharge_summaries['subject_id'].unique()
# tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
# model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct",
#                                              device_map="auto",
#                                             #  attn_implementation="flash_attention_2",
#                                             #  torch_dtype=torch.float16
#                                              )
# generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# prompt_templates = [
#     (
#         "You are a clinician summarizing a patient's discharge summary. "
#         "Do not include your tasks or instructions.\n"
#         "Do not include name, date, or other identifying information.\n"
#         "Summarize the following text in one paragraph for the main theme: {concept}\n"
#         "Text: {context}\n"
#         "Summary::"
#     ),
#     (
#         "You are a clinician summarizing a patient's discharge summary. "
#         "Do not include your tasks or instructions.\n"
#         "Do not include name, date, or other identifying information.\n"
#         "Summarize the following text in one paragraph for the main theme: {concept}\n"
#         "Mention if any of the following are present: {expanded_concepts}\n"
#         "Text: {context}\n"
#         "Summary::"
#     ),
# ]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# text = "The patient was admitted for pneumonia."
# query_result = embeddings.embed_query(text)
# print(query_result)
CHROMA_PATH = "/tmp/chroma"
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

def trim_after_last_period(sentence):
    # Find the last occurrence of a period
    last_period_index = sentence.rfind('.')

    # If a period is found, trim the sentence after the last period
    if last_period_index != -1:
        return re.sub('\d', '*',sentence[:last_period_index + 1])
    else:
        # If no period is found, return the original sentence
        return re.sub('\d', '*',sentence)

def split_text(subject_id):
    docs  = []
    for index, notes in discharge_summaries[discharge_summaries['subject_id'] == subject_id].iterrows():
        discharge_note = notes['text']
        concept = discharge_summaries[discharge_summaries['subject_id'] == subject_id]['concept'].values[0]
        expanded_concepts = str(discharge_summaries[discharge_summaries['subject_id'] == subject_id]['expanded_concepts'].values[0])
        print(f"Subject ID: {subject_id}, Concept: {concept}, Expanded Concepts: {expanded_concepts}")
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=64, chunk_overlap=5
        )
        split_docs = text_splitter.split_text(discharge_note)
        for doc in split_docs:
            docs.append(
                Document(
                    page_content=doc,
                    metadata={"subject_id": str(subject_id), "concept": concept, "expanded_concepts": expanded_concepts},
                )
            )
    return docs

def save_to_chroma(docs, subject_id):
    db = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=CHROMA_PATH + f"/{subject_id}",
    )
    # db.persist()
    # print(f"Saved to {CHROMA_PATH + f'/{subject_id}'}")
    return db

summaries = []
for subject_id in tqdm.tqdm(subject_ids[:10]):  # Limit to 10 summaries
    docs = split_text(subject_id)
    db = save_to_chroma(docs, subject_id)
    concept = discharge_summaries[discharge_summaries['subject_id'] == subject_id]['concept'].values[0]
    expanded_concepts = discharge_summaries[discharge_summaries['subject_id'] == subject_id]['expanded_concepts'].values[0]

    summary = ["", ""]

    results = db.similarity_search(concept, k=5)
    context_text = "\n\n --- \n\n".join([result.page_content for result in results])
    summary[0] = context_text
    # if len(context_text) > 2500:
    #     print("Context text is too long. Skipping...")
    #     continue
    # for response in generator(
    #     prompt_templates[0].format(concept=concept, context=context_text), max_new_tokens=256
    # ):
    #     print("Default Summary: \n")
    #     full_response = response["generated_text"].split("::")[-1]
    #     summary = full_response.split("##")[0].strip() + "\n" # remove the system prompt that gets appended
    #     print(trim_after_last_period(summary))
    #     print("---------------------------------------------------------------\n\n")
    results = db.similarity_search(concept + " " + expanded_concepts, k=5)
    context_text = "\n\n --- \n\n".join([result.page_content for result in results])
    summary[1] = context_text
    # if len(context_text) > 2500:
    #     print("Context text is too long. Skipping...")
    #     continue
    # for response in generator(
    #     prompt_templates[1].format(concept=concept, context=context_text, expanded_concepts=expanded_concepts), max_new_tokens=256
    # ):
    #     print("Expanded Summary: \n")
    #     full_response = response["generated_text"].split("::")[-1]
    #     summary = full_response.split("##")[0].strip() + "\n" # remove the system prompt that gets appended
    #     print(trim_after_last_period(summary))
    #     print("--------------------------------------------------------------- \n\n")
    summaries.append([subject_id, summary[0], summary[1], concept])
df = pd.DataFrame(summaries, columns=['subject_id', 'text', 'jog_memory', 'concept'])
df.to_csv('~/data/mapped_summaries.csv', index=False)

