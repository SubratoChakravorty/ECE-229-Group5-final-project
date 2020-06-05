# Quick Start

## Setup
Now our dashboard is supported by python3 >= 3.7, you can create a new virtual environment by anaconda

```bash
$ conda env create -n dash -f environment.yml && conda activate dash && pip3 install --editable . 
```

After installing the dependencies, you can start the dashboard by

```bash
$ python3 src/ui/dashboard.py
```

Then you can find the page on `http://127.0.0.1:8050/`

## Test

You can also run test procedures by

```bash
$ pytest --headless
```
## Build the docs

To rebuild the docs supported by sphinx, you can edit the files under docs/source and then 

```bash
$ cd docs && sphinx-build -b html source build && make html
```

Then you can find your documentation at docs/build/html/index.html