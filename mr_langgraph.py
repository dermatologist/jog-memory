import operator
from typing import Annotated, List, TypedDict
import asyncio
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
import pandas as pd
import torch
model_id = "microsoft/Phi-3-mini-128k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=16)
llm = HuggingFacePipeline(pipeline=pipe)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

map_template = "Write a concise summary of the following: {context}."

reduce_template = """
The following is a set of summaries:
{docs}
Take these and distill it into a final, consolidated summary
of the main themes.
"""

map_prompt = ChatPromptTemplate([("human", map_template)])
reduce_prompt = ChatPromptTemplate([("human", reduce_template)])

map_chain = map_prompt | llm | StrOutputParser()
reduce_chain = reduce_prompt | llm | StrOutputParser()

# Graph components: define the components that will make up the graph


# This will be the overall state of the main graph.
# It will contain the input document contents, corresponding
# summaries, and a final summary.
class OverallState(TypedDict):
    # Notice here we use the operator.add
    # This is because we want combine all the summaries we generate
    # from individual nodes back into one list - this is essentially
    # the "reduce" part
    contents: List[str]
    summaries: Annotated[list, operator.add]
    final_summary: str


# This will be the state of the node that we will "map" all
# documents to in order to generate summaries
class SummaryState(TypedDict):
    content: str


# Here we generate a summary, given a document
async def generate_summary(state: SummaryState):
    response = await map_chain.ainvoke(state["content"])
    return {"summaries": [response]}


# Here we define the logic to map out over the documents
# We will use this an edge in the graph
def map_summaries(state: OverallState):
    # We will return a list of `Send` objects
    # Each `Send` object consists of the name of a node in the graph
    # as well as the state to send to that node
    return [
        Send("generate_summary", {"content": content}) for content in state["contents"]
    ]


# Here we will generate the final summary
async def generate_final_summary(state: OverallState):
    response = await reduce_chain.ainvoke(state["summaries"])
    return {"final_summary": response}


# Construct the graph: here we put everything together to construct our graph
graph = StateGraph(OverallState)
graph.add_node("generate_summary", generate_summary)
graph.add_node("generate_final_summary", generate_final_summary)
graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
graph.add_edge("generate_summary", "generate_final_summary")
graph.add_edge("generate_final_summary", END)
app = graph.compile()


async def main(documents):
    # Call the graph:
    async for step in app.astream({"contents": [doc.page_content for doc in documents]}):
        print(step)

discharge_summaries = pd.read_csv('data/discharge_sample.csv')
subject_id = 0
documents = []
for index, row in discharge_summaries.iterrows():
    if subject_id != row['subject_id']:
        subject_id = row['subject_id']
        documents = []
    print(row['subject_id'])
    print("--------------------")
    user_input = row['text'].replace('\n', ' ').strip()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=2500, chunk_overlap=8
    )
    split_docs = text_splitter.split_text(user_input)
    print(f"Generated {len(split_docs)} documents.")
    for doc in split_docs:
        documents.append(Document(page_content=doc, metadata={"subject_id": row['subject_id']}))
    print("--------------------")
    torch.cuda.empty_cache()
    asyncio.run(main(documents))
