# python -m pip install llama-cpp-python --prefer-binary --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/basic/cu122


from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
import pandas as pd
from tqdm import tqdm
import re
import random
import os

discharge_summaries = pd.read_csv('~/data/map2_summaries.csv')
main_concepts = pd.read_csv('~/data/main_concepts.csv')
subject_ids = discharge_summaries['subject_id'].unique()

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
    n_ctx=2048,  # The context window size
    max_tokens=256,  # The maximum number of tokens to generate
    temperature=0.3,  # The temperature for sampling
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

simple_template = """
You are a clinician summarizing a patient's discharge summary.
Do not include your tasks or instructions.
Do not include name, date, or other identifying information.
Context: {prompt}
Summarize the above context in one paragraph for the theme: {concept}
Summary:"""

expanded_template = """
You are a clinician summarizing a patient's discharge summary.
Do not include your tasks or instructions.
Do not include name, date, or other identifying information.
Context: {prompt}
Summarize the above context in one paragraph for the themes: {expanded_concepts}, {concept}
Summary:"""

simple_prompt = PromptTemplate.from_template(simple_template)
simple_llm_chain = simple_prompt | llm
expanded_prompt = PromptTemplate.from_template(expanded_template)
expanded_llm_chain = expanded_prompt | llm

def trim_after_last_period(sentence):
    # Find the last occurrence of a period
    last_period_index = sentence.rfind('.')

    # If a period is found, trim the sentence after the last period
    if last_period_index != -1:
        return re.sub('\d', '*',sentence[:last_period_index + 1])
    else:
        # If no period is found, return the original sentence
        return re.sub('\d', '*',sentence)

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

for subject_id in tqdm(subject_ids):
    x = discharge_summaries[discharge_summaries['subject_id'] == subject_id]
    print(x["concept"].values[0])
    print("text")
    print("====================================================")
    print(x["text"].values[0])
    print("jog_memory")
    print("====================================================")
    print(x["jog_memory"].values[0])

# # step 4: reduce the discharge summaries
# summaries = []
# print("Reducing discharge summaries...")
# for subject_id in tqdm(subject_ids):
#     concept = discharge_summaries[discharge_summaries['subject_id'] == subject_id]['concept'].values[0]
#     expanded_concepts = str(main_concepts[main_concepts['subject_id'] == subject_id]['expanded_concepts'].values[0])
#     simple_notes = discharge_summaries[discharge_summaries['subject_id'] == subject_id]['text'].values[0]
#     expanded_notes = discharge_summaries[discharge_summaries['subject_id'] == subject_id]['jog_memory'].values[0]

#     print(f"Subject ID: {subject_id}, Concept: {concept}, Expanded Concepts: {expanded_concepts}, Simple Notes: {simple_notes}, Expanded Notes: {expanded_notes}")
#     simple_summary = simple_llm_chain.invoke({"prompt": simple_notes, "concept": concept})
#     expanded_summary = expanded_llm_chain.invoke({"prompt": expanded_notes, "concept": concept, "expanded_concepts": expanded_concepts})

#     summaries.append([subject_id, simple_summary, expanded_summary, concept])
#     content = f"""
#     Subject ID: {subject_id} | Concept: {concept} | Expanded Concepts: {expanded_concepts}

#     Default Summary:
#     {trim_after_last_period(simple_summary)}

#     JOG Memory Summary:
#     {trim_after_last_period(expanded_summary)}

#     ---------------------------------------------------------------
#     """
#     with open(os.environ['HOME'] + '/data/report.txt', 'a') as f:
#         f.write(content)

#     with open(os.environ['HOME'] + '/data/report_anon.txt', 'a') as f:
#         f.write(anonymize([simple_summary, expanded_summary], concept))

# df = pd.DataFrame(summaries, columns=['subject_id', 'text', 'jog_memory', 'concept'])
# df.to_csv('~/data/final_summaries.csv', index=False)