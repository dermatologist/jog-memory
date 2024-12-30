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

import click
import requests
from pypdf import PdfReader

from . import JogMemory, JogRag
from .log import suppress_stdout_stderr

try:
    from . import __version__
except ImportError:
    __version__ = "unknown"


@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Will print verbose messages.")
@click.option("--file", "-f", help="Text file to summarize.")
@click.option("output", "--output", "-o", help="Output file to save the summary.")
@click.option(
    "--theme",
    "-t",
    multiple=False,
    default="auto",
    help="Theme for summarization (default: auto)",
)
@click.option(
    "--n_ctx", "-c", multiple=False, default=2048 + 256, help="Context window size"
)
@click.option(
    "--max_tokens", "-m", multiple=False, default=256, help="Max tokens to generate"
)
@click.option(
    "--k", "-k", multiple=False, default=5, help="Number of documents to retrieve"
)
@click.option(
    "--n_gpu_layers", "-g", multiple=False, default=-1, help="Number of GPU layers"
)
@click.option("--expand", "-e", is_flag=True, help="Expand the summary")
@click.option(
    "--llm", "-l", help="LLM model (default: mradermacher/Llama3-Med42-8B-GGUF)"
)
@click.option(
    "--embedding",
    "-b",
    help="Embedding file (default: garyw/clinical-embeddings-100d-w2v-cr)",
)
@click.option("--clear", "-x", is_flag=True, help="Clear the text")
def cli(
    verbose,
    file,
    output,
    theme,
    n_ctx,
    max_tokens,
    k,
    n_gpu_layers,
    expand,
    llm,
    embedding,
    clear,
):
    if file:  # if file is provided, read the file
        # if file is pdf
        if file.endswith(".pdf"):
            pdf = PdfReader(file)
            text = ""
            for page in pdf.pages:
                text = text + " " + page.extract_text()
        # if file is url
        elif file.startswith("http"):
            response = requests.get(file)
            text = response.text
        else:
            with open(file, "r") as f:
                text = f.read()
    else:
        text = click.prompt("Enter text to summarize", type=str)
    n_ctx = n_ctx
    max_tokens = max_tokens
    k = k
    n_gpu_layers = n_gpu_layers

    jog_memory = JogMemory(
        model_path=llm,
        embedding_path=embedding,
        n_ctx=n_ctx,
        max_tokens=max_tokens,
        n_gpu_layers=n_gpu_layers,
    )
    jog_rag = JogRag(
        n_ctx=n_ctx,
    )
    if clear:
        jog_memory.clear_text()
    jog_memory.append_text(text)
    if theme == "auto":
        theme = jog_memory.find_concept(text[: n_ctx - 300])
        click.secho(f"Theme: {theme}\n", fg="yellow")
    if expand:
        expanded = jog_memory.expand_concept(theme)
        click.secho(f"Expanded concepts: {expanded}\n", fg="yellow")
    else:
        expanded = ""
    # RAG if length of text exceeds context window size
    if len(jog_memory.get_text()) > (n_ctx - 300):
        docs = jog_rag.split_text(
            jog_memory.get_text(), "subject", theme, expanded, k=k
        )
        context = jog_rag.get_context(theme, expanded, k=k)
    else:
        context = jog_memory.get_text()
    click.secho(f"Summarizing: ....\n", fg="yellow")
    if verbose:
        click.secho(f"Context: {context}\n", fg="yellow")
        summary = jog_memory.summarize(context, theme, expanded)
    else:
        with suppress_stdout_stderr():
            summary = jog_memory.summarize(context, theme, expanded)
    click.secho(f"Summary: {summary}\n", fg="green")
    if output:
        with open(output, "w") as f:
            f.write(summary)


def main_routine():
    click.echo("_________________________________________")
    click.secho("Jog Memory Summary:  working: ......", fg="green")
    cli()  # run the main function
    click.echo("Summarization complete.")


if __name__ == "__main__":
    main_routine()
