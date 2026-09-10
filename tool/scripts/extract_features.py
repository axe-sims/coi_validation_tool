"""
=============================================================================
Sequence Feature and Taxonomy Extraction 
=============================================================================
This script processes nucleotide sequences from a compressed FASTA file and 
extracts a comprehensive set of numerical features for model training. 

Calculated features include:
- Basic metrics: Sequence length, individual base frequencies (A, C, G, T)
- Composition biases: GC Content, GC Skew, AT Skew, Purine/Pyrimidine Ratio
- K-mer frequencies: Normalized counts for all 2-mers, 3-mers, and 4-mers

Basic usage (for label 1 sequences):
    python extract_features.py -i Balanced_Train_Positives.fasta.gz \
        -t Train_Positives_Taxa.csv.gz \
        --out_features Feat_Train_Pos.csv.gz \
        --out_taxonomy Taxa_Train_Pos.csv.gz \
        -l 1

"""

import csv
import itertools
import argparse
import gzip
import pandas as pd
from collections import Counter
from Bio import SeqIO

# k-mer generation for 2, 3, and 4-mers
bases = ['A', 'C', 'G', 'T']
kmers_2 = [''.join(p) for p in itertools.product(bases, repeat=2)]
kmers_3 = [''.join(p) for p in itertools.product(bases, repeat=3)]
kmers_4 = [''.join(p) for p in itertools.product(bases, repeat=4)]
all_kmers = kmers_2 + kmers_3 + kmers_4

def clean_str(s):
    """Strip carriage returns, newlines, and excess whitespace to prevent CSV corruption."""
    return str(s).strip().replace('\r', '').replace('\n', '')

def process_sequence(header, seq_raw, label=""):
    seq = seq_raw.upper().replace("U", "T")
    seq = "".join(b for b in seq if b in bases)
    length = len(seq)
    
    if length < 10: return None
    
    counts = Counter(seq)
    a, c, g, t = counts["A"], counts["C"], counts["G"], counts["T"]
    
    features = {
        "Header": clean_str(header), "Label": label, "Length": length,
        "A_freq": a/length if length else 0, 
        "C_freq": c/length if length else 0, 
        "G_freq": g/length if length else 0, 
        "T_freq": t/length if length else 0,
        "GC_Content": (g+c)/length if length else 0, 
        "GC_Skew": (g-c)/(g+c) if g+c else 0,
        "AT_Skew": (a-t)/(a+t) if a+t else 0, 
        "Pur_Pyr_Ratio": (a+g)/(c+t) if c+t else 0
    }
    
    kmer_counts = Counter()
    for k in [2, 3, 4]:
        for i in range(length-k+1): kmer_counts[seq[i:i+k]] += 1
            
    for kmer in all_kmers:
        denom = length-len(kmer)+1
        features[kmer] = kmer_counts[kmer]/denom if denom > 0 else 0
        
    return features

# main function to run the extraction process
def run_extraction(input_fasta, out_features, out_taxonomy=None, taxa_dicts=None, label=""):
    if taxa_dicts is None:
        taxa_dicts = []
        
    taxa_lookup = {}
    
    # building a single taxonomy dictionary from multiple input files (if provided)
    if taxa_dicts:
        for t_file in taxa_dicts:
            df = pd.read_csv(t_file, compression='gzip', low_memory=False)
            df['Header'] = df['Header'].astype(str).str.strip().str.replace('\r', '').str.replace('\n', '')
            
            # drop duplicates to ensure unique headers in the dictionary
            df = df.drop_duplicates(subset=['Header'])
            
            taxa_lookup.update(df.set_index('Header').to_dict('index'))

    feature_headers = ["Header", "Label", "Length", "A_freq", "C_freq", "G_freq", "T_freq", "GC_Content", "GC_Skew", "AT_Skew", "Pur_Pyr_Ratio"] + all_kmers
    taxonomy_headers = ["Header", "Phylum", "Class", "Order", "Family", "Genus", "Species"]

    with gzip.open(input_fasta, "rt") as fasta, \
         gzip.open(out_features, "wt", newline="") as fout:

        fw = csv.DictWriter(fout, fieldnames=feature_headers)
        fw.writeheader()
        
        # only open and prepare the taxonomy file if asked
        tout = gzip.open(out_taxonomy, "wt", newline="") if out_taxonomy else None
        if tout:
            tw = csv.DictWriter(tout, fieldnames=taxonomy_headers)
            tw.writeheader()

        for record in SeqIO.parse(fasta, "fasta"):
            feats = process_sequence(record.id, str(record.seq), label)
            
            if feats:
                clean_header = clean_str(record.id)
                fw.writerow(feats)
                
                # only write taxonomy data if the file is open
                if tout:
                    if clean_header in taxa_lookup:
                        meta = taxa_lookup[clean_header]
                        taxa = [clean_str(meta.get(k, "Unknown")) for k in taxonomy_headers[1:]]
                    else:
                        taxa = ["Unknown"] * 6

                    tw.writerow({
                        "Header": clean_header,
                        "Phylum": taxa[0], "Class": taxa[1], "Order": taxa[2],
                        "Family": taxa[3], "Genus": taxa[4], "Species": taxa[5]
                    })
        
        if tout:
            tout.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract DNA features using an optional taxonomy dictionary.")
    parser.add_argument("-i", "--input", required=True, help="Input FASTA.gz file")
    parser.add_argument("--out_features", required=True, help="Output Features CSV.gz file")
    parser.add_argument("--out_taxonomy", required=False, default=None, help="Output Taxonomy CSV.gz file (optional)")
    parser.add_argument("-t", "--taxa_dicts", nargs="*", required=False, default=[], help="One or more Taxa.csv.gz files to use as dictionary (optional)")
    parser.add_argument("-l", "--label", default="", help="Class label (optional)")
    
    args = parser.parse_args()
    run_extraction(args.input, args.out_features, args.out_taxonomy, args.taxa_dicts, args.label)