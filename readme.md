# Imageflow
This project is part of my MA-Thesis, it builds 
upon the idea of the voxelmorph framework, but tries 
to implement flow
field generation via normalized flows.

Note: you will need a working installation of current pytorch, which is not explicitly installed by the setup.py since what exact version you want might be dependent on your system.

To build it locally clone the git repository:
```
git clone https://github.com/theIcebaer/imageflow.git
```
Then install the project as local development package:
```
cd repo
pip install -e .
```
Note that there is a known issue with pycharm and local dev packages. Im not sure yet how to resolve this best, but one way is to tell pycharm to directly manipulate the PYTHONPATH environment variable.
Scripts for unsupervised and supervised training can be found in /scritps.
