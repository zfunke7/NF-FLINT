# NF-FLINT
Build-your-own CR3BP normal forms all in Python. Accelerated by python-flint. 

## Installation
Download as .zip and extract into a local directory of your choice, or `git clone` into this repository. From this project's base directory, you can install all dependencies with:
`pip install -r requirements.txt`

### Package Dependencies
* python-flint==0.8.0
* matplotlib==3.8.4
* numpy==2.3.3
* scipy==1.16.2

You may also wish to `pip install jupyter` to open and use the interactive Tutorial. 

## What's new?
This project offers a fast computer algebra system (CAS) that you can use without ever leaving Python. This is implemented via the `Poly` and `RealPoly` classes, which use Python-wrapped FLINT C routines under-the-hood for polynomial arithmetic but also allow for `dict` and `array` representations, as well as fast evaluation with `numpy`.  

Functionally, this means that you can execute all of the machinery associated with the creation and manipulation of Hamiltonian normal forms and their associated generating functions and coordinate transformations all from one place. 

## Do I have to build my own normal forms to use this?
No, the `data/` directory provides the necessary components to perform both numerical and analytical transformations between synodic Cartesian coordinates and the associated normal-form canonical coordinates (or action-angle coordinates, or any intermediate transformation for that matter) for truncation orders $N=11$ and $N=22$. To learn how to use these, just skip ahead in the Tutorial. 