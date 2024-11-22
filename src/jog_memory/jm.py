# python -m pip install llama-cpp-python --prefer-binary --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/basic/cu122
from huggingface_hub import hf_hub_download, snapshot_download
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from gensim.models import Word2Vec
import re
import tempfile
import logging
logging.getLogger("langchain_text_splitters.base").setLevel(logging.ERROR)

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
                 summary_prompt = None,
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
        self.summary_prompt = summary_prompt
        REPO_ID = "garyw/clinical-embeddings-100d-w2v-cr"
        FILENAME = "w2v_OA_CR_100d.bin"
        self.word_embedding = Word2Vec.load(snapshot_download(repo_id=REPO_ID)+"/"+FILENAME)
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
        # if _concept is not a list, convert it to a list
        if not isinstance(concept, list):
            concepts = [concept]
        else:
            concepts = concept
        words = []
        phrases = []
        for concept in concepts:
            print(f"Concept: {concept}")
            try:
                words.extend(self.word_embedding.wv.most_similar(concept, topn=10))
            except:
                pass
        for (word, score) in words:
            print(f"Word: {word}, Score: {score}")
            if score > 0.75:
                phrases.append(word.replace("_", " "))
        return phrases

    def trim_after_last_period(self, sentence):
        # Find the last occurrence of a period
        last_period_index = sentence.rfind('.')

        # If a period is found, trim the sentence after the last period
        if last_period_index != -1:
            return re.sub('\d', '*',sentence[:last_period_index + 1])
        else:
            # If no period is found, return the original sentence
            return re.sub('\d', '*',sentence)

    def get_theme_prompt(self):
        if self.theme_prompt:
            return self.theme_prompt
        return """
        Identify the main diagnosis and/or procedure in the text in one or two words.
        DO NOT include anything else in the response.
        Examples:
        Text: The patient was admitted for pneumonia.
        Main concept: pneumonia
        Text: The patient underwent hernia repair.
        Main concept: hernia repair
        Text: {prompt}
        Main concept: """

    def get_summary_prompt(self):
        if self.summary_prompt:
            return self.summary_prompt
        return """
        You are a clinician summarizing a patient's discharge summary.
        Do not include your tasks or instructions.
        Do not include name, date, or other identifying information.
        Context: {prompt}
        Summarize the above context in one paragraph for the theme(s): {concept} {expanded_concepts}
        Summary:"""

    def clear_concept(self):
        self.concept = None
        self.expanded_concepts = []

    def set_concept(self, concept):
        self.concept = concept

    def get_concept(self):
        return self.concept

    def find_concept(self):
        prompt = PromptTemplate.from_template(self.get_theme_prompt())
        llm_chain = prompt | self.llm
        text = self.get_text()[:self.n_ctx - 300]
        output = llm_chain.invoke({"prompt": text})
        self.clear_concept()
        self.set_concept(output.split('\n', 1)[0].strip())
        return self.get_concept()

    def summarize(self, concept="", expanded_concepts=[]):
        prompt = PromptTemplate.from_template(self.get_summary_prompt())
        llm_chain = prompt | self.llm
        text = self.get_text()
        if len(text) > self.n_ctx:
            text = text[:self.n_ctx - 300]
            print("Text trimmed to fit context window. Please split text into smaller chunks or use RAG.")
        text = self.get_text()[:self.n_ctx - 300]
        output = llm_chain.invoke({"prompt": text, "concept": concept, "expanded_concepts": str(expanded_concepts)})
        return self.trim_after_last_period(output)

    def __str__(self):
        return str(self.get_text())