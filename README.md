<div align="center">

## Quantized jet taggers

</div>


Based on https://github.com/heidelberg-hepml/lloca-experiments

### 1. Getting started

Clone the repository.

```bash
git clone git@github.com:heidelberg-hepml/tagger-quantization.git
cd tagger-quantization
```

Install the repository in a virtual environment

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

### 2. Pre-commit hooks

Install the pre-commit hook once after dependencies are installed:

```bash
pre-commit install
```

Run the checks manually with:

```bash
pre-commit run --all-files
```

It will run automatically at each commit to apply black formatting.