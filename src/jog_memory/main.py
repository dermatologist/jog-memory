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
def cli(verbose, file, theme):
    if verbose:
        print("verbose")



def main_routine():
    click.echo("_________________________________________")
    click.echo("Jog Memory Summary v " + __version__ + " working:.....")
    cli()  # run the main function
    click.echo("Summarization complete.")


if __name__ == '__main__':
    main_routine()