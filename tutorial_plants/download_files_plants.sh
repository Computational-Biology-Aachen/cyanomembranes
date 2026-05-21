#!/bin/bash

# Create directory to store PDB files
mkdir -p data_plants

OMP="https://biomembhub.org/shared/opm-assets/pdb"
PDBTM="https://pdbtm.unitmp.org/api/v1/entry"

# Download files from databases
wget -q --show-progress -O "data_plants/3JCU-PSII-spinach.pdb" "${OMP}/3jcu.pdb"
wget -q --show-progress -O "data_plants/6RQF-Cyt-spinach.pdb" "${OMP}/6rqf.pdb"
wget -q --show-progress -O "data_plants/8IWX-LHCII-spinach.trpdb" "${PDBTM}/8iwx.trpdb"
wget -q --show-progress -O "data_plants/7E0K-LHCII-chlamy.pdb" "${OMP}/7e0k.pdb"
