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
import re
import tempfile

import click
from gensim.models import Word2Vec
from huggingface_hub import hf_hub_download, snapshot_download

# python -m pip install llama-cpp-python --prefer-binary --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/basic/cu122
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

logging.getLogger("langchain_text_splitters.base").setLevel(logging.ERROR)
from .log import suppress_stdout_stderr

logger = logging.getLogger(__name__)


class JogMemory:

    def __init__(
        self,
        model_path: str = None,  # Use the default model if None
        embedding_path: str = None,  # Use the default embedding if None
        n_gpu_layers=-1,  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU. 0 for all on CPU.
        n_batch=256,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
        n_ctx=2048,  # The context window size
        max_tokens=128,  # The maximum number of tokens to generate
        temperature=0.1,  # The temperature for sampling
        theme_prompt=None,  # The theme prompt
        summary_prompt=None,  # The summary prompt
        expanded_prompt=None,  # The expanded prompt
    ):
        self.tempfile = tempfile.SpooledTemporaryFile(mode="a+t", max_size=10000)
        self.n_gpu_layers = n_gpu_layers
        self.n_batch = n_batch
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        self.temperature = temperature
        if model_path is None:
            logger.info(f"Downloading the default model.")
            self.model_path = hf_hub_download(
                "mradermacher/Llama3-Med42-8B-GGUF",
                filename="Llama3-Med42-8B.Q8_0.gguf",
            )
        else:
            logger.info(f"Loading model from {model_path}.")
            self.model_path = model_path
        # Callbacks support token-wise streaming
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        with suppress_stdout_stderr():
            self.llm = LlamaCpp(
                model_path=self.model_path,  # Download the model file first
                n_gpu_layers=self.n_gpu_layers,  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU. 0 for all on CPU.
                n_batch=self.n_batch,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
                n_ctx=self.n_ctx,  # The context window size
                max_tokens=self.max_tokens,  # The maximum number of tokens to generate
                temperature=self.temperature,  # The temperature for sampling
                callback_manager=self.callback_manager,  # The callback manager
                verbose=True,  # Verbose is required to pass to the callback manager
            )
            self.theme_prompt = theme_prompt
            self.summary_prompt = summary_prompt
            self.expanded_prompt = expanded_prompt
            if embedding_path is None:
                REPO_ID = "garyw/clinical-embeddings-100d-w2v-cr"
                FILENAME = "w2v_OA_CR_100d.bin"
                logger.info(f"Loading default embedding.")
                self.word_embedding = Word2Vec.load(
                    snapshot_download(repo_id=REPO_ID) + "/" + FILENAME
                )
            else:
                logger.info(f"Loading embedding from {embedding_path}.")
                self.word_embedding = Word2Vec.load(embedding_path)
            self.concept = None

    def append_text(self, text):
        self.tempfile.write(text)

    def get_text(self):
        self.tempfile.seek(0)
        return self.tempfile.read()

    def clear_text(self):
        self.tempfile.seek(0)
        self.tempfile.truncate()

    def expand_concept(self, concept):
        concept = concept.lower().strip().replace(" ", "_")
        # if _concept is not a list, convert it to a list
        if not isinstance(concept, list):
            concepts = [concept]
        else:
            concepts = concept
        words = []
        phrases = []
        for concept in concepts:
            click.secho(f"Concept: {concept}", fg="yellow")
            try:
                words.extend(self.word_embedding.wv.most_similar(concept, topn=10))
            except:
                pass
        for word, score in words:
            click.secho(f"Word: {word}, Score: {score}", fg="bright_blue")
            if score > 0.75:
                phrases.append(word.replace("_", " "))
        return phrases

    def trim_after_last_period(self, sentence):
        # Find the last occurrence of a period
        last_period_index = sentence.rfind(".")
        # If a period is found, trim the sentence after the last period
        if last_period_index != -1:
            return re.sub(r"\d", "*", sentence[: last_period_index + 1])
        else:
            # If no period is found, return the original sentence
            return re.sub(r"\d", "*", sentence)

    def get_theme_prompt(self):
        if self.theme_prompt:
            return self.theme_prompt
        return """
        Identify ONLY the Main concept in the text in one or two words.
        Examples:
        Text: The patient was admitted for pneumonia.
        Main concept: pneumonia
        Text: The patient underwent hernia repair.
        Main concept: hernia repair
        Text: {prompt}
        Main concept:"""

    def get_summary_prompt(self):
        if self.summary_prompt:
            return self.summary_prompt
        return """
        You are a clinician summarizing a patient's clinical note.
        Do not repeat information or add information that is not in the context.
        Summarize the below context in a single paragraph for the theme: {concept}
        Context: {prompt}
        Summary:"""

    def get_expanded_prompt(self):
        if self.expanded_prompt:
            return self.expanded_prompt
        return """
        You are a clinician summarizing a patient's clinical note.
        Do not repeat information or add information that is not in the context.
        Comment on {expanded_concepts} IF they are present in the context.
        Summarize the below context in a single paragraph for the theme: {concept}
        Context: {prompt}
        Summary:"""

    def clear_concept(self):
        self.concept = None
        self.expanded_concepts = []

    def set_concept(self, concept):
        self.concept = concept

    def get_concept(self):
        return self.concept

    def find_concept(self, text=None):
        prompt = PromptTemplate.from_template(self.get_theme_prompt())
        llm_chain = prompt | self.llm
        if text is None:
            text = self.get_text()[: self.n_ctx - 300]
        output = llm_chain.invoke({"prompt": text})
        self.clear_concept()
        output = re.split(r"\n|,", output)
        self.set_concept(output[0].strip())
        return self.get_concept()

    def summarize(self, text=None, concept="", expanded_concepts=[]):
        prompt = PromptTemplate.from_template(self.get_summary_prompt())
        expanded_prompt = PromptTemplate.from_template(self.get_expanded_prompt())
        llm_chain = prompt | self.llm
        expanded_chain = expanded_prompt | self.llm
        if text is None:
            text = self.get_text()
        original_length = len(text)
        if original_length > self.n_ctx:
            text = text[: self.n_ctx - 30]
            click.echo(f"Text length: {original_length}, Trimmed length: {len(text)}\n")
        else:
            click.echo(f"Text length: {original_length}\n")
        if len(expanded_concepts) > 0:
            output = expanded_chain.invoke(
                {
                    "prompt": text,
                    "concept": concept,
                    "expanded_concepts": ", ".join(expanded_concepts),
                }
            )
        else:
            output = llm_chain.invoke({"prompt": text, "concept": concept})
        return self.trim_after_last_period(output)

    def __str__(self):
        return str(self.get_text())
