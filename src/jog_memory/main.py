import click
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
@click.option('--max_tokens', '-m', multiple=False, default=128 + 128, help='Max tokens to generate')
@click.option('--k', '-k', multiple=False, default=5, help='Number of documents to retrieve')
@click.option('--n_gpu_layers', '-g', multiple=False, default=-1, help='Number of GPU layers')
def cli(verbose, file, theme, n_ctx, max_tokens, k, n_gpu_layers):
    if verbose:
        print("verbose")
    if file: # if file is provided, read the file
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
    if theme == 'auto':
        theme = jog_memory.find_concept(text)

def main_routine():
    click.echo("_________________________________________")
    click.echo("Jog Memory Summary v " + __version__ + " working:.....")
    cli()  # run the main function
    click.echo("Summarization complete.")


if __name__ == '__main__':
    main_routine()