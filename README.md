# diffaux

Auxiliary Modeling for Diffsky Galaxies

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

[![PyPI](https://img.shields.io/pypi/v/diffaux?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/diffaux/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ArgonneCPAC/diffaux/smoke-test.yml)](https://github.com/ArgonneCPAC/diffaux/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/ArgonneCPAC/diffaux/branch/main/graph/badge.svg)](https://codecov.io/gh/ArgonneCPAC/diffaux)
[![Read The Docs](https://img.shields.io/readthedocs/diffaux)](https://diffaux.readthedocs.io/)
[![Benchmarks](https://img.shields.io/github/actions/workflow/status/ArgonneCPAC/diffaux/asv-main.yml?label=benchmarks)](https://ArgonneCPAC.github.io/diffaux/)

This project was automatically generated using the LINCC-Frameworks 
[python-project-template](https://github.com/lincc-frameworks/python-project-template).


## Galaxy Size Modeling

A description of the data and methods used for writing the code in this repository
can be found [here](https://docs.google.com/document/d/1NYZ8EAOqi89DrQoV9Oz1EI30lIsl56kHoXQ3qygtc74/edit?usp=sharing)

Documentation for the code used to generate the galaxy size model can be found
[here](https://docs.google.com/document/d/1NYZ8EAOqi89DrQoV9Oz1EI30lIsl56kHoXQ3qygtc74/edit?usp=sharing)

Documentation for the code used to generate the disk-bulge decomposition can be found
[here[(https://docs.google.com/document/d/1664LHkPow2wl8rEKda8m_qomO-i1XnnihesBAQN6Mec/edit?usp=sharing)

## Dev Guide - Getting Started

Before installing any dependencies or writing code, it's a great idea to create a
virtual environment. LINCC-Frameworks engineers primarily use `conda` to manage virtual
environments. If you have conda installed locally, you can run the following to
create and activate a new environment.

```
>> conda create env -n <env_name> python=3.10
>> conda activate <env_name>
```


Once you have created a new environment, you can install this project for local
development using the following commands:

```
>> pip install -e .'[dev]'
>> pre-commit install
>> conda install pandoc
```

Notes:
1. The single quotes around `'[dev]'` may not be required for your operating system.
2. `pre-commit install` will initialize pre-commit for this local repository, so
   that a set of tests will be run prior to completing a local commit. For more
   information, see the Python Project Template documentation on 
   [pre-commit](https://lincc-ppt.readthedocs.io/en/latest/practices/precommit.html)
3. Install `pandoc` allows you to verify that automatic rendering of Jupyter notebooks
   into documentation for ReadTheDocs works as expected. For more information, see
   the Python Project Template documentation on
   [Sphinx and Python Notebooks](https://lincc-ppt.readthedocs.io/en/latest/practices/sphinx.html#python-notebooks)
