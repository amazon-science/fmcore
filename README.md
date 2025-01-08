## FMCore
**fmcore** is a specialized toolkit for scaling experimental research with Foundation Models. Our utilities for inference, tuning, and evaluation handle billion-scale models and datasets, letting you skip the boilerplate and focus on pushing AI research forward. Developed by researchers, for researchers, **fmcore** offers a robust, flexible platform that streamlines your workflow without sacrificing rigor.

## Installation

We recommend installing dependencies in a new Anaconda environment using the commands below.

These commands were tested to work on `Deep Learning AMI GPU PyTorch 1.13.1 (Amazon Linux 2) 20230221` on AWS.

Install dependencies:

```commandline
conda create -n fmcore python=3.11.8 --yes  
conda activate fmcore
pip install uv   ## Speed up installation
uv pip install -r requirements.txt
uv pip install "spacy==3.7.4" "spacy-transformers==1.3.5"
uv pip install "setuptools==69.5.1"
```

Certain features need additional steps, e.g. 
```commandline
python -m spacy download en_core_web_lg
python -c "import nltk; nltk.download('punkt');"
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

