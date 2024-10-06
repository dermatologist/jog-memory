import pandas as pd
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import time
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch


discharge_summaries = pd.read_csv('data/discharge_sample.csv')
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", device_map="auto")
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
    (
        "You are a clinician summarizing a patient's discharge summary. "
        "Summarize the following text in one paragraph.\n"
        "Text: {prompt}\n"
        "Summary::"
    ),
]


# Print the first 5 rows of the DataFrame
# print(discharge_summaries.head())

for index, row in discharge_summaries.iterrows():
    print(row['subject_id'])
    print("--------------------")
    user_input = row['text'].replace('\n', ' ').strip()
    user_input = user_input[:3000]  # limit the length of the input to 500 characters
    responses = generator(
        [template.format(prompt=user_input) for template in prompt_templates], max_new_tokens=256
    )
    for idx, response in enumerate(responses):
        print(f"Response to Template #{idx}:")
        reply = response[0]["generated_text"].split("::")[-1].strip()
        line = reply.split("\n")[0]
        print(line + "\n\n\n")
    print("--------------------")