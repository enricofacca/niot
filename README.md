# Description
Network Inpainting via Optimal Transport (NIOT) is a python package for reconstructing corrupted networks based on optimal (branched) transport principles. See [^1] for all details (preprint available [here](https://arxiv.org/abs/2405.13520)).

# Installation
The code is written in python and it is based on [Firedrake](https://www.firedrakeproject.org). Check [the dedicated page](https://www.firedrakeproject.org/download.html) for installation guidance. Consider also installing using
```
firedrake-install --doi 10.5281/zenodo.11242360
```
to ensure having the same software used in the manuscript [^1].


Clone this repository with
```
git clone https://github.com/enricofacca/niot.git
``` 
move in the niot directory, and install the niot package with
```
pip install -e .
```

# Examples
Move in the directory examples/2024FaccaNordbottenHanson and follow the instruction to reproduce the results of the paper.

# Authors
Enrico Facca, enrico.facca@uib.no : Departement of Mathematics, University of Bergen, Bergen, Norway.

[^1]:"Network Inpainting via Optimal Transport" Enrico Facca, Jan Martin Nordbotten, Erik Andreas Hanson