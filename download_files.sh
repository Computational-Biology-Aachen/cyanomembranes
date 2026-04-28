#!/bin/bash

# Create directory to store PDB files
mkdir -p data_cyano

OMP="https://biomembhub.org/shared/opm-assets/pdb"
PDBTM="https://pdbtm.unitmp.org/api/v1/entry"

# Download files from databases
wget -q --show-progress -O "data_cyano/1JB0-PSI-syn-cocc.pdb" "${OMP}/1jb0.pdb"
wget -q --show-progress -O "data_cyano/1NEK-SDH-Ecoli.pdb" "${OMP}/1nek.pdb"
wget -q --show-progress -O "data_cyano/3WU2-PSII-ThermosynVul.pdb" "${OMP}/3wu2.pdb"
wget -q --show-progress -O "data_cyano/4H13-cytb6f.trpdb" "${PDBTM}/4h13.trpdb"
wget -q --show-progress -O "data_cyano/1OCO-cytoxidase-bov.trpdb" "${PDBTM}/1oco.trpdb"
wget -q --show-progress -O "data_cyano/1xl4-Kchannel-Pmagnetotacticum.trpdb" "${PDBTM}/1xl4.trpdb"
wget -q --show-progress -O "data_cyano/4HEA-NDH1-thermo.trpdb" "${PDBTM}/4hea.trpdb"

