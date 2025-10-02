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

## Project status
Currently, everything is set up only for the CR3BP L1 point. Eventually, support will be added for the other equilibrium points. 

## Acknowledgements and references
Special thanks to David Schwab (Penn State '24) for his insight, support, and wonderfully instructive PhD dissertation as well as Carson Hunsberger (Penn State) for making his NF code available, which should be referenced especially if interested in resonant NF, stationkeeping, and support for other collinear equilibrium points (https://github.com/CarsonHunsberger/NF_CR3BP_Python). Thanks also to Luke Peterson (CU Boulder '24, UT Austin) for his insight, advice, and useful publications. His toolkit can be found here: https://github.com/lukethomaspeterson/Toolkit 

* Peterson, Luke, and D. Scheeres. 2023. “Local Orbital Elements for the Circular Restricted Three-Body Problem.” Journal of Guidance, Control, and Dynamics 46 (August). https://doi.org/10.2514/1.G007435.
* Peterson, Luke T., and Daniel J. Scheeres. 2024. “Gauss Equations for Local Action-Angle Orbital Elements in Cislunar Space.” Journal of Guidance, Control, and Dynamics 47 (11): 2273–86. https://doi.org/10.2514/1.G008399.
* Schwab, David. 2023. “Cislunar Transport Characterization for Space Situational Awareness.” PhD Dissertation, The Pennsylvania State University. https://etda.libraries.psu.edu/catalog/21986dvs5558.
* Hunsberger, Carson, David Schwab, Roshan Eapen, and Puneet Singla. 2025. “Comparing Normal Form Representations for Station-Keeping Near Cislunar Libration Points.” Paper presented at AAS/AIAA Space Flight Mechanics Meeting, Kaua’i, HI. January 22. https://www.space-flight.org/docs/2025_winter/2025_winter.html.
* Almanza-Soto, Juan-Pablo, and Kathleen Howell. 2025. “Persistence of Restricted Three-Body Problem Normal Form Structures in Higher-Fidelity Models.” Paper presented at AAS/AIAA Astrodynamics Specialist Conference, Boston, MA. August 11. https://www.researchgate.net/publication/394467666_Persistence_of_Restricted_Three-Body_Problem_Normal_Form_Structures_in_Higher-Fidelity_Models.
* Jorba, Àngel. 1999. “A Methodology for the Numerical Computation of Normal Forms, Centre Manifolds and First Integrals of Hamiltonian Systems.” Experimental Mathematics 8 (2): 155–95. https://doi.org/10.1080/10586458.1999.10504397.
