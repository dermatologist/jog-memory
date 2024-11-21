# python -m pip install llama-cpp-python --prefer-binary --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/basic/cu122
from huggingface_hub import hf_hub_download, snapshot_download
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec
import re
import tempfile


class JogMemory:

    def __init__(self,
                 model_name: str = "mradermacher/Llama3-Med42-8B-GGUF",
                 model_file: str = "Llama3-Med42-8B.Q8_0.gguf",
                 n_gpu_layers = -1, # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU. 0 for all on CPU.
                 n_batch = 256, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
                 n_ctx = 2048, # The context window size
                 max_tokens = 128, # The maximum number of tokens to generate
                 temperature = 0.1, # The temperature for sampling
                 theme_prompt = None,
                 ):
        self.tempfile = tempfile.SpooledTemporaryFile(mode='a+t', max_size=10000)
        self.model_name = model_name
        self.model_file = model_file
        self.n_gpu_layers = n_gpu_layers
        self.n_batch = n_batch
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_path = hf_hub_download(self.model_name, filename=self.model_file)
        # Callbacks support token-wise streaming
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.llm = LlamaCpp(
            model_path=self.model_path,  # Download the model file first
            n_gpu_layers=self.n_gpu_layers,
            n_batch=self.n_batch,
            n_ctx=self.n_ctx,  # The context window size
            max_tokens=self.max_tokens,  # The maximum number of tokens to generate
            temperature=self.temperature,  # The temperature for sampling
            callback_manager=self.callback_manager,
            verbose=True,  # Verbose is required to pass to the callback manager
        )
        self.theme_prompt = theme_prompt

    def append_text(self, text):
        self.tempfile.write(text)

    def get_text(self):
        self.tempfile.seek(0)
        return self.tempfile.read()

    def clear_text(self):
        self.tempfile.seek(0)
        self.tempfile.truncate()

    def get_theme_prompt(self):
        if self.theme_prompt:
            return self.theme_prompt
        return """
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

    def __str__(self):
        return str(self.get_text())