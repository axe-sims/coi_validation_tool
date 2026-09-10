# coi_validation_tool

A command-line interface (CLI) tool for validating Cytochrome c Oxidase I (COI) gene sequences from an input FASTA file of your target sequences. It automates compositional feature extraction and performs neural network inference to accurately distinguish functional COI sequences from non-target sequences.

## Repository Structure

```text
coi_validation_tool/
├── scripts/
│   ├── extract_features.py
│   └── predict_new_data.py
├── models/
│   ├── NN_deployment_scaler.pkl
│   ├── NN_expected_cols.pkl
│   └── Best_NN_Model.pkl
├── environment.yml
└── classify_coi.sh
```

## 1. Preparing Environment

```bash
conda env create -f environment.yml
conda activate coxi_pipeline
```

## 2. Make Bash Script Executable

```bash
chmod +x classify_coi.sh
```

## 3. Run Tool

Run the tool inside the `coi_validation_tool/` directory:

```bash
./classify_coi.sh <input_fasta.gz> <output_predictions.csv.gz>
```
