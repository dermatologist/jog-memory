<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/jog-memory.svg?branch=main)](https://cirrus-ci.com/github/<USER>/jog-memory)
[![ReadTheDocs](https://readthedocs.org/projects/jog-memory/badge/?version=latest)](https://jog-memory.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/jog-memory/main.svg)](https://coveralls.io/r/<USER>/jog-memory)
[![PyPI-Server](https://img.shields.io/pypi/v/jog-memory.svg)](https://pypi.org/project/jog-memory/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/jog-memory.svg)](https://anaconda.org/conda-forge/jog-memory)
[![Monthly Downloads](https://pepy.tech/badge/jog-memory/month)](https://pepy.tech/project/jog-memory)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/jog-memory)
-->

# jog-memory

> Jog Memory is a proof of concept package for domain adapted summarization of clinical documents

`jog-memory` is a Python package designed to summarize clinical documents using LLMs (Large Language Models) and word embeddings for domain adaptation. More to come after review.

## Installation

You can install `jog-memory` directly from GitHub using pip:

```sh
pip install git+https://github.com/dermatologist/jog-memory.git
```

## Usage

After installing the package, you can use the command-line interface (CLI) to summarize clinical documents with concept expansion using LLMs and
custom trained word embeddings. Use [`train.py`](train.py) to train your own word embeddings.

Below are some examples of how to use the CLI.


### Summarize a PDF File

To summarize a PDF file, use the `--file` option with a PDF file and the --theme option. If no theme is provided, the theme will be automatically detected from first few lines of the document.

```sh
jms --file path/to/your/file.pdf --theme psoriasis
```

### Save the Summary to an Output File

To save the summary to an output file, use the `--output` option:

```sh
jms --file path/to/your/file.txt --output path/to/output.txt
```

### Expand the Summary

To expand the summary with additional concepts, use the `--expand` option:

```sh
jms --file path/to/your/file.txt --expand
```

### Additional Options

- `--verbose` or `-v`: Print verbose messages.
- `--theme` or `-t`: Theme for summarization (default: auto).
- `--n_ctx` or `-c`: Context window size (default: 2304).
- `--max_tokens` or `-m`: Max tokens to generate (default: 256).
- `--k` or `-k`: Number of documents to retrieve (default: 5).
- `--n_gpu_layers` or `-g`: Number of GPU layers (default: -1).
- `--llm` or `-l`: LLM model (default: mradermacher/Llama3-Med42-8B-GGUF).
- `--embedding` or `-b`: Embedding file (default: garyw/clinical-embeddings-100d-w2v-cr).
- `--clear` or `-x`: Clear the text.

## Contributing

Contributions are welcome! Please see the [`CONTRIBUTING.md`](CONTRIBUTING.md ) file for more information.

## Citation

Details on how to cite this work will be provided after review.

## Give us a star ⭐️
If you find this project useful, give us a star. It helps others discover the project.

## Contributors

* [Bell Eapen](https://nuchange.ca) | [![Twitter Follow](https://img.shields.io/twitter/follow/beapen?style=social)](https://twitter.com/beapen)
* PRs welcome. See CONTRIBUTING.md

