import accelerate
import pandas as pd
from gensim.models import Word2Vec
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import re
from tqdm import tqdm

df = pd.read_csv('~/data/discharge_5000.csv')
sample = df.sample(n=500) # 30
# unique_values = sample['subject_id'].value_counts()
# subject_id = unique_values[unique_values == 2].index # <3
# subject_id.append(unique_values[unique_values == 3].index)

subject_id = sample['subject_id'].unique()

subject_id = subject_id[0:25]
print(f"Subject IDs: {subject_id}")
discharge_summaries = df[df['subject_id'].isin(subject_id)]
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

REPO_ID = "garyw/clinical-embeddings-100d-w2v-cr"
FILENAME = "w2v_OA_CR_100d.bin"
word_embedding = Word2Vec.load(snapshot_download(repo_id=REPO_ID)+"/"+FILENAME)

def expand_concept(_concept):
    # if _concept is not a list, convert it to a list
    if not isinstance(_concept, list):
        concepts = [_concept]
    else:
        concepts = _concept
    _words = []
    phrases = []
    for concept in concepts:
        print(f"Concept: {concept}")
        try:
            _words.extend(word_embedding.wv.most_similar(concept, topn=10))
        except:
            pass
    for (word, score) in _words:
        print(f"Word: {word}, Score: {score}")
        if score > 0.75:
            phrases.append(word.replace("_", " "))
    return phrases

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
# excluded = [10000826, 10000032, 10000980, 10001186, 10038849]
print("Identifying main concepts in discharge summaries...")
# for index, row in discharge_summaries.iterrows():
for index, row in tqdm(discharge_summaries.iterrows(), total=discharge_summaries.shape[0]):
        if subject_id != row['subject_id']:
            subject_id = row['subject_id']
            # if subject_id in excluded:
            #     continue
            concept = ""
            discharge_note = row['text'][:2500]
            response = generator(
                prompt_templates[0].format(prompt=discharge_note), max_new_tokens=256
            )
            concept = response[0]["generated_text"].split("::")[-1].strip().split("\n")[0]
            expanded_concepts = expand_concept(concept.lower().strip().replace(" ", "_"))
            expanded_concepts = re.sub(r'\W+', ' ', str(expanded_concepts)).strip()
            print(f"Subject ID: {subject_id}, Main Concept: {concept}, Expanded Concepts: {expanded_concepts}")
            if len(expanded_concepts) > 5:
                main_concepts.append([subject_id, concept, expanded_concepts, row["text"]])
df = pd.DataFrame(main_concepts, columns=['subject_id', 'concept', 'expanded_concepts', 'text'])
df.to_csv('~/data/main_concepts.csv', index=False)


