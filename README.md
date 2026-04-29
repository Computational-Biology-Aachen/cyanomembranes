# Cyanomembranes

> Investigating effects of spatial organization in cyanobacterial thylakoid membranes. Repository to reproduce figures of **Mesoscale crowding and microdomain formation positively influence plastoquinone diffusion in simulated cyanobacteria thylakoid membrane**

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Data](#data)
- [Reproducing the Figures](#reproducing-the-figures)
- [Methods Summary](#methods-summary)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Overview

Spatial heterogeneity in cyanobacterial thylakoid membranes has been observed. However, it is not fully clear how spatial organization forms and what purpose it serves. This project uses computational membranes and lattice diffusion to investigate how the formation of pigment-protein microdomains and crystalline arrays affects the movement within the cyanobacterial thylakoid membrane.
---

## Repository Structure

```
.
├── download_files.sh     # Download protein structures from database
├── membrane_analysis
│   ├── config.py
│   ├── __init__.py
│   ├── pipeline.py
│   ├── plotting.py
│   └── utils.py
├── notebooks
│   ├── analysis.ipynb         # Execute model scenarios, save output and create figures
│   ├── crystal_analysis.ipynb 
│   ├── __init__.py
│   ├── make_figures.ipynb     # Use analysis figures to create pub figures
│   ├── make_membranes.ipynb   # Create computational membranes
│   ├── plot_membranes.ipynb
│   ├── scenarios.py           # Define model scenarios
│   └── test_notebook.ipynb
├── pixi.lock
├── pyproject.toml
├── README.md
├── src
│   └── cyanomembranes
│       ├── brownian_lattice.py
│       ├── brownian.py
│       ├── geo_algorithms.py
│       ├── geo_utils.py
│       ├── __init__.py
│       ├── pdb_utils.py
│       └── utils.py
└── tutorial
    └── brownian_diffusion.ipynb

```

---

## Getting Started

### Prerequisites

- [Pixi](https://prefix.dev/docs/pixi/overview) — handles all dependencies.

### Installation

```bash
git clone https://github.com/Computational-Biology-Aachen/cyanomembranes.git
cd cyanomembranes
pixi install
```

To launch a Jupyter environment:

```bash
pixi run jupyter lab
```

or run in IDE

---

## Data

Protein structure files are downloaded from OMP (https://opm.phar.umich.edu/) or TMDPB database (https://pdbtm.unitmp.org/entry/3wu2)

**Download instructions** (if data is not bundled):

```bash
# Example: download with curl or a dedicated script
bash src/download_data.sh
```

---

## Reproducing the Figures

Run the notebooks **in order**. Each notebook lists its inputs and outputs.

### Step 1 — Membrane creation (`notebooks/make_membranes.ipynb`)

What it does:
- Creates in silico membranes across different scenarios
- Saves membranes as wkt file in output directory
- Attention: Can take a long time depending on machine and available cores

---

### Step 2 — Membrane Figures (`notebooks/plot_membranes.ipynb`)

What it does:
- Generates example figures of in silico membranes 
- Performs RDF and Channel width analysis

---

### Step 3 — Analysis (`notebooks/analysis.ipynb `)

What it does:
- Runs simulations based on scenarios defined in scenario.py
- Saves results in output directory 
- Attention: Can take a long time depending on machine and available cores
- Warning: Takes a lot of RAM. For machines wi RAM lower than 64 GB it is highly recommended to increase the save interval in the membrane_analysis/config.py file

---

### Step 4 — Create publication figures (`notebooks/make_figures.ipynb `)

What it does:
- Assembles publication figures based on analysis output

---



## Methods Summary

- Formation of computational thylakoid membranes: No-fit polygons and compression 
- Electron carrier movement: Lattice diffusion

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{AuthorYear,
  author  = {Author, A. and Author, B.},
  title   = {Paper title},
  journal = {Journal Name},
  year    = {2025},
  doi     = {10.xxxx/xxxxx}
}
```

---

## License

Code: [](LICENSE)  
Data: [](https://creativecommons.org/licenses/by/4.0/) *(adjust as appropriate)*

---

## Contact

**Corresponding author:** Your Name — your.email@institution.de  
Feel free to open a [GitHub issue](https://github.com/your-org/your-repo/issues) for questions about the code.