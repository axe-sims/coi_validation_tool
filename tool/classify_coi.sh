#!/bin/bash
set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: ./classify_coi.sh <input_fasta.gz> <output_predictions.csv.gz>"
    exit 1
fi

INPUT_FASTA=$1
OUTPUT_PREDS=$2
TEMP_FEAT=$(mktemp --suffix=_features.csv.gz)
TEMP_TAXA=$(mktemp --suffix=_taxa.csv.gz)

echo "Extracting compositional features..."
python scripts/extract_features.py \
    -i "${INPUT_FASTA}" \
    --out_features "${TEMP_FEAT}" \
    --out_taxonomy "${TEMP_TAXA}"

echo "Running Neural Network inference..."
python scripts/predict_new_data.py \
    --new_data "${TEMP_FEAT}" \
    --scaler models/NN_deployment_scaler.pkl \
    --cols models/NN_expected_cols.pkl \
    --model models/Best_NN_Model.pkl \
    --output "${OUTPUT_PREDS}"

rm -f "${TEMP_FEAT}" "${TEMP_TAXA}"
echo "Done. Results saved to ${OUTPUT_PREDS}"
