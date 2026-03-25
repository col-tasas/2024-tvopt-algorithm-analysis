<!-- PROJECT SHIELDS -->
[![arXiv][arxiv-shield]][arxiv-url]
[![MIT License][license-shield]][license-url]
[![ReseachGate][researchgate-shield]][researchgate-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
[![GIT][git-shield]][git-url]

# A Linear Parameter-Varying Framework for the Analysis of Time-Varying Optimization Algorithms

This repository contains the code from our paper

> Jakob, Fabian, and Andrea Iannelli. "A Linear Parameter-Varying Framework for the Analysis of Time-Varying Optimization Algorithms." arXiv preprint. arXiv:2501.07461 (2025).

## Installation

The repo has been developed and tested with Python 3.11 and the dependencies given in `requirements.txt`

```bash
pip install -r requirements.txt
```

You may also use open source SDP solvers such as `CVXOPT`. However, the numeric results are most robust using `Mosek`, which can be used with an academic license:

```bash
pip install Mosek
```

## Notebooks

Demonstrates the use of the repo; reproduces the plots in the paper.

`tracking_certificates.ipynb`: Algorithm certificates across problem parameters

`tracking_bounds.ipynb`: Certified tracking error bounds for a given objective and algorithm

## Repository Structure

```
src/
├── tracking_certificates.ipynb
├── tracking_bounds.ipynb
├── lib/
│   ├── algorithms/
│   │   ├── unconstrained.py
│   │   └── constrained.py
│   ├── analysis/
│   │   ├── lure.py
│   │   ├── lyapunov.py
│   │   ├── polytope.py
│   │   ├── solver.py
│   │   └── run_solver.py
│   ├── simulation/
│   │   ├── algorithm.py
│   │   ├── objectives.py
│   │   └── simulate.py
│   └── utils/
│       └── plot.py
└── tests/
```

## Running Tests

```bash
pytest src/tests/
```

Tests cover algorithms, objectives, IQC builders, Lyapunov matrices, polytope
utilities, and simulation outputs (89 tests). Tests that invoke the MOSEK/CVXPY
solver are marked `slow` and skipped by default; run them with `pytest -m slow`.

## Contact

Fabian Jakob — [fabian.jakob@ist.uni-stuttgart.de](mailto:fabian.jakob@ist.uni-stuttgart.de)

[git-shield]: https://img.shields.io/badge/Github-fjakob-white?logo=github
[git-url]: https://github.com/fjakob
[license-shield]: https://img.shields.io/badge/License-MIT-T?style=flat&color=blue
[license-url]: https://github.com/col-tasas/2024-tvopt-algorithm-analysis/blob/main/LICENSE
[arxiv-shield]: https://img.shields.io/badge/arXiv-2501.07461-t?style=flat&logo=arxiv&logoColor=white&color=red
[arxiv-url]: https://arxiv.org/abs/2501.07461
[researchgate-shield]: https://img.shields.io/badge/ResearchGate-Fabian%20Jakob-T?style=flat&logo=researchgate&color=darkgreen
[researchgate-url]: https://www.researchgate.net/profile/Fabian-Jakob-4
[linkedin-shield]: https://img.shields.io/badge/Linkedin-Fabian%20Jakob-T?style=flat&logo=linkedin&logoColor=blue&color=blue
[linkedin-url]: https://www.linkedin.com/in/fabian-jakob/
