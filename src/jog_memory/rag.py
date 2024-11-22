from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document
import os
import shutil
import logging
logging.getLogger("langchain_text_splitters.base").setLevel(logging.ERROR)


class JogRag:

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.CHROMA_PATH = "/tmp/chroma"
        self.db = None

    def save_to_chroma(self, docs, subject_id):
        if os.path.exists(self.CHROMA_PATH):
            shutil.rmtree(self.CHROMA_PATH)
        db = Chroma.from_documents(
            docs,
            self.embeddings,
            persist_directory=self.CHROMA_PATH + f"/{subject_id}",
        )
        # db.persist()
        # print(f"Saved to {CHROMA_PATH + f'/{subject_id}'}")
        self.db = db
        return db

    def split_text(self, text, subject_id, concept="", expanded_concepts=[], chunk_size=64, chunk_overlap=5, save_to_chroma=True):
        docs = []
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        split_docs = text_splitter.split_text(text)
        print(f"Split {len(text)} long text into {len(docs)} documents.\n")
        for doc in split_docs:
            docs.append(
                Document(
                    page_content=doc,
                    metadata={"subject_id": str(subject_id), "concept": concept, "expanded_concepts": str(expanded_concepts)},
                )
            )
        if save_to_chroma:
            self.save_to_chroma(docs, subject_id)
        return docs

    def get_context(self, concept="", expanded_concepts=[], k=5):
        if self.db is None:
            return "No database found. Please run split_text() first."
        results = self.db.similarity_search(concept + " " + " ".join(expanded_concepts), k=k)
        context_text = "\n\n --- \n\n".join([result.page_content for result in results])
        return context_text