![F M Core logo](img/logos/aws-like-wide-gradient.png)

`fmcore` is a specialized toolkit that empowers AI scientists to break new ground by simplifying large-scale experimentation with massive Foundation Models and datasets.

A primary bottleneck in Foundation Model research is implementation overhead. With `fmcore`, scientists can rapidly prototype new innovations in hours instead of weeks, accelerating the path to new research breakthroughs or user experiences.

Key features:
- Easy scaling of model training and inference (see examples).
- Standardized interfaces for parameter tuning and evaluation.
- Built-in support for distributed computing and Foundation Model parallelism.

## Installation

The minimal `fmcore` package can be installed from PyPI:

```
pip install fmcore 
```

To get all features, we recommend installing in a new Conda environment:

```commandline
conda create -n fmcore python=3.11 --yes
conda activate fmcore
pip install uv
uv pip install "fmcore[all]"
```

## License

This project is licensed under the Apache-2.0 License.

## Contributing to `fmcore`

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.
