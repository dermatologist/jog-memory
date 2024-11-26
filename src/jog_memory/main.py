import click
from pypdf import PdfReader
from . import JogMemory
from . import JogRag
try:
    from . import __version__
except ImportError:
    __version__ = "unknown"

@click.command()
@click.option('--verbose', '-v', is_flag=True, help="Will print verbose messages.")
@click.option('--file', '-f', is_flag=True, help="Text file to summarize.")
@click.option('--theme', '-t', multiple=False, default='auto',
              help='Theme for summarization (default: auto)')
@click.option('--n_ctx', '-c', multiple=False, default=2048 + 256, help='Context window size')
@click.option('--max_tokens', '-m', multiple=False, default=256, help='Max tokens to generate')
@click.option('--k', '-k', multiple=False, default=5, help='Number of documents to retrieve')
@click.option('--n_gpu_layers', '-g', multiple=False, default=-1, help='Number of GPU layers')
@click.option('--expand', '-e', is_flag=True, help="Expand the summary")
@click.option('--clear', '-x', is_flag=True, help="Clear the text")
def cli(verbose, file, theme, n_ctx, max_tokens, k, n_gpu_layers, expand, clear):
    if verbose:
        click.echo("verbose")
    if file: # if file is provided, read the file
        # if file is pdf
        if file.endswith('.pdf'):
            pdf = PdfReader(file)
            text = ' '.join([page.text for page in pdf.pages])
        else:
            with open(file, 'r') as f:
                text = f.read()
    else:
        text = click.prompt("Enter text to summarize", type=str)
    n_ctx = n_ctx
    max_tokens = max_tokens
    k=k
    n_gpu_layers = n_gpu_layers

    jog_memory = JogMemory(
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
    if theme == 'auto':
        theme = jog_memory.find_concept(text)
        click.echo(f"Theme: {theme}")
    if expand:
        expanded = jog_memory.expand_concept(theme)
        click.echo(f"Expanded concepts: {expanded}")
    else:
        expanded = ""
    # RAG if length of text exceeds context window size
    if len(jog_memory.get_text()) > (n_ctx-300):
        docs = jog_rag.split_text(jog_memory.get_text(), "subject", theme, expanded, k=k)
        context = jog_rag.get_context(theme, expanded, k=k)
    else:
        context = jog_memory.get_text()
    click.secho(f"Summary: {jog_memory.summarize(context, theme, expanded)}\n", fg='green')

def main_routine():
    click.echo("_________________________________________")
    click.echo("Jog Memory Summary v " + __version__ + " working:.....")
    cli()  # run the main function
    click.echo("Summarization complete.")


if __name__ == '__main__':
    main_routine()