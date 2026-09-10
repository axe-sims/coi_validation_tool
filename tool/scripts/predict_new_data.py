"""
=============================================================================
COI Sequence Prediction and Inference 
=============================================================================
This script applies a pre-trained machine learning model, deployment scaler, 
and feature column list to new nucleotide feature datasets to predict 
functional Cytochrome C Oxidase I (COI) sequences.

Basic usage:
    python predict.py \
        --new_data data/inference/Unseen_Features.csv.gz \
        --scaler models/Ensemble_deployment_scaler.pkl \
        --cols models/Ensemble_expected_cols.pkl \
        --model models/Ensemble_Stacking_Model.pkl \
        --output results/inference/Unseen_Predictions.csv.gz

"""

import argparse
import pandas as pd
import joblib
import os
import gzip

def predict_on_new_data(new_features_csv, scaler_pkl, cols_pkl, model_pkl, output_csv):
    """
    Predicts COI using pre-trained model and scaler.
    """
    
    # loading saved data
    print("Loading saved data...")
    model = joblib.load(model_pkl)
    scaler = joblib.load(scaler_pkl)
    expected_cols = joblib.load(cols_pkl) 
    scaler_cols = list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else expected_cols
    
    is_gzipped = new_features_csv.endswith('.gz')
    
    skip_rows = 0
    write_mode = 'w'
    write_header = True
    total_processed_previously = 0

    if os.path.exists(output_csv):
        open_func = gzip.open if output_csv.endswith('.gz') else open
        mode = 'rt' if output_csv.endswith('.gz') else 'r'
        
        with open_func(output_csv, mode) as f:
            lines = sum(1 for _ in f)
        
        if lines > 1:
            total_processed_previously = lines - 1
            skip_rows = total_processed_previously 
            write_mode = 'a'
            write_header = False
            print(f"Resuming after {total_processed_previously:,} sequences...")

    batch_size = 500000
    batch_num = 1
    
    print(f"Starting batch prediction from {new_features_csv}...")

    # batch processing
    for chunk in pd.read_csv(new_features_csv, chunksize=batch_size, skiprows=skip_rows):
        
        # ensure all columns the scaler saw during EDA exist
        for col in scaler_cols:
            if col not in chunk.columns:
                chunk[col] = 0.0

        # scale the data
        X_chunk_scaled = scaler.transform(chunk[scaler_cols])
        
        X_scaled_df = pd.DataFrame(X_chunk_scaled, columns=scaler_cols)
        
        # extract only the columns the model was trained on 
        for col in expected_cols:
            if col not in X_scaled_df.columns:
                X_scaled_df[col] = 0.0
                
        X_model_ready = X_scaled_df[expected_cols]

        # predict using the filtered columns
        predictions = model.predict(X_model_ready)
        probabilities = model.predict_proba(X_model_ready)[:, 1]
        
        # only create true label column if it exists in the input 
        results_dict = {'Header': chunk['Header']}
        
        if 'Label' in chunk.columns:
            results_dict['True_Label'] = chunk['Label']
            
        results_dict['Predicted_Label'] = predictions
        results_dict['COI_Probability'] = probabilities
        
        results_chunk = pd.DataFrame(results_dict)
        
        current_compression = 'gzip' if output_csv.endswith('.gz') else None
        
        results_chunk.to_csv(
            output_csv, 
            mode=write_mode if batch_num == 1 else 'a', 
            index=False, 
            header=write_header if batch_num == 1 else False,
            compression=current_compression
        )
        
        chunk_total = len(results_chunk)
        print(f"Batch {batch_num:<2} processed | COI: {predictions.sum():>7,} | Total: {chunk_total:>7,}")
        batch_num += 1

    print(f"Success. Results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COI Prediction Tool")
    parser.add_argument("--new_data", required=True, help="Input features (CSV or CSV.GZ)")
    parser.add_argument("--scaler", required=True, help="Path to pre-trained scaler.pkl")
    parser.add_argument("--cols", required=True, help="Path to pre-trained expected_cols.pkl")
    parser.add_argument("--model", required=True, help="Path to trained model.pkl")
    parser.add_argument("--output", required=True, help="Output predictions (CSV or CSV.GZ)")
    
    args = parser.parse_args()
    
    predict_on_new_data(args.new_data, args.scaler, args.cols, args.model, args.output)