from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "microsoft/Phi-3-mini-128k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=64)
llm = HuggingFacePipeline(pipeline=pipe)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter

from langchain_core.documents import Document
import pandas as pd



# documents = [
#     Document(page_content="Apples are red", metadata={"title": "apple_book"}),
#     Document(page_content="Blueberries are blue", metadata={"title": "blueberry_book"}),
#     Document(page_content="Bananas are yelow", metadata={"title": "banana_book"}),
# ]

# Map
map_template = "Write a concise summary of the following: {docs}."
map_prompt = ChatPromptTemplate([("human", map_template)])
map_chain = LLMChain(llm=llm, prompt=map_prompt)


# Reduce
reduce_template = """
The following is a set of summaries:
{docs}
Take these and distill it into a final, consolidated summary
of the main themes.
"""
reduce_prompt = ChatPromptTemplate([("human", reduce_template)])
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)


# Takes a list of documents, combines them into a single string, and passes this to an LLMChain
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="docs"
)

# Combines and iteratively reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    # This is final chain that is called.
    combine_documents_chain=combine_documents_chain,
    # If documents exceed context for `StuffDocumentsChain`
    collapse_documents_chain=combine_documents_chain,
    # The maximum number of tokens to group documents into.
    token_max=1000,
)

# Combining documents by mapping a chain over them, then combining results
map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name="docs",
    # Return the results of the map steps in the output
    return_intermediate_steps=False,
)



discharge_summaries = pd.read_csv('data/discharge_sample.csv')
for index, row in discharge_summaries.iterrows():
    documents = []
    print(row['subject_id'])
    print("--------------------")
    user_input = row['text']
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=750, chunk_overlap=10
    )
    split_docs = text_splitter.split_text(user_input)
    print(f"Generated {len(split_docs)} documents.")
    for doc in split_docs:
        documents.append(Document(page_content=doc, metadata={"subject_id": row['subject_id']}))
    print("--------------------")
    result = map_reduce_chain.invoke(documents)

    print(result["output_text"])
