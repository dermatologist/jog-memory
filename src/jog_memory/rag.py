"""
 Copyright 2024 Bell Eapen

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging
import os
import shutil

import click
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.getLogger("langchain_text_splitters.base").setLevel(logging.ERROR)


class JogRag:

    def __init__(self, n_ctx=None):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.CHROMA_PATH = "/tmp/chroma"
        self.db = None
        self.n_ctx = n_ctx

    def save_to_chroma(self, docs, subject_id):
        if os.path.exists(self.CHROMA_PATH):
            shutil.rmtree(self.CHROMA_PATH)
        db = Chroma.from_documents(
            docs,
            self.embeddings,
            persist_directory=self.CHROMA_PATH + f"/{subject_id}",
        )
        self.db = db
        return db

    def split_text(
        self,
        text,
        subject_id,
        concept="",
        expanded_concepts=[],
        chunk_size=64,
        chunk_overlap=5,
        k=5,
        save_to_chroma=True,
    ):
        docs = []
        if self.n_ctx:
            chunk_size = int(self.n_ctx / (k * 5))
            click.secho(f"Chunk size: {chunk_size} \n", fg="red")
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        split_docs = text_splitter.split_text(text)
        click.secho(
            f"Split {len(text)} long text into {len(split_docs)} documents.\n", fg="red"
        )
        for doc in split_docs:
            docs.append(
                Document(
                    page_content=doc,
                    metadata={
                        "subject_id": str(subject_id),
                        "concept": concept,
                        "expanded_concepts": str(expanded_concepts),
                    },
                )
            )
        if save_to_chroma:
            self.save_to_chroma(docs, subject_id)
        return docs

    def get_context(self, concept="", expanded_concepts=[], k=5):
        if self.db is None:
            return "No database found. Please run split_text() first."
        results = self.db.similarity_search(
            concept + " " + " ".join(expanded_concepts), k=k
        )
        context_text = "\n\n --- \n\n".join([result.page_content for result in results])
        return context_text
