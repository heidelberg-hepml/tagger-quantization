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
pre-commit install
```

The pre-commit hook will automatically run on each commit. You can also run it manually with ```pre-commit run --all-files```.
