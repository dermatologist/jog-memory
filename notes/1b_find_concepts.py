# python -m pip install llama-cpp-python --prefer-binary --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/basic/cu122


from huggingface_hub import hf_hub_download, snapshot_download
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec
import re

model_name = "mradermacher/Llama3-Med42-8B-GGUF"
model_file = "Llama3-Med42-8B.Q8_0.gguf"

model_path = hf_hub_download(model_name, filename=model_file)

n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU. 0 for all on CPU.
n_batch = 256  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path=model_path,  # Download the model file first
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=1024,  # The context window size
    max_tokens=128,  # The maximum number of tokens to generate
    temperature=0.1,  # The temperature for sampling
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

template = """
You are a clinician reviewing a patient's discharge summary.
Do not include your tasks or instructions.
Identify the main diagnosis and/or procedure in the text in one or two words.
Examples:
Text: The patient was admitted for pneumonia.
Main concept: pneumonia
Text: The patient underwent hernia repair.
Main concept: hernia repair
Text: {prompt}
Main concept: """

prompt = PromptTemplate.from_template(template)
llm_chain = prompt | llm

df = pd.read_csv('~/data/discharge_5000.csv')
sample = df.sample(n=500) # 30

subject_id = sample['subject_id'].unique()

subject_id = subject_id[0:25]
print(f"Subject IDs: {subject_id}")
discharge_summaries = df[df['subject_id'].isin(subject_id)]

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
            # response = generator(
            #     prompt_templates[0].format(prompt=discharge_note), max_new_tokens=256
            # )
            concept = llm_chain.invoke({"prompt": discharge_note})
            expanded_concepts = expand_concept(concept.lower().strip().replace(" ", "_"))
            expanded_concepts = re.sub(r'\W+', ' ', str(expanded_concepts)).strip()
            print(f"Subject ID: {subject_id}, Main Concept: {concept}, Expanded Concepts: {expanded_concepts}")
            if len(expanded_concepts) > 5:
                main_concepts.append([subject_id, concept, expanded_concepts, row["text"]])
df = pd.DataFrame(main_concepts, columns=['subject_id', 'concept', 'expanded_concepts', 'text'])
df.to_csv('~/data/main_concepts.csv', index=False)



