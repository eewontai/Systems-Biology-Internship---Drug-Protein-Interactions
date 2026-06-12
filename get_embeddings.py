#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 13:49:16 2025

@author: eta

With gpu - docker & batching!

Multi-GPU Optimized ChemBERTa + ProtBERT Pipeline for IC50 Prediction

Generate 10 datasets and merge them into one - consisting of drug cls tokens, protein cls tokens, and ic50 values
"""

import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import torch
from rdkit import Chem
import gc

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

print(f"Using {n_gpu} GPU(s)")

# Load tokenizers
chembert_tokenizer = AutoTokenizer.from_pretrained("/models/chemberta")
protbert_tokenizer = BertTokenizer.from_pretrained("/models/protbert", do_lower_case=False)

# Load models
chembert_model = AutoModel.from_pretrained("/models/chemberta")
protbert_model = BertModel.from_pretrained("/models/protbert")

# Wrap with DataParallel if using multiple GPUs
if n_gpu > 1:
    chembert_model = torch.nn.DataParallel(chembert_model)
    protbert_model = torch.nn.DataParallel(protbert_model)

chembert_model = chembert_model.to(device).eval()
protbert_model = protbert_model.to(device).eval()


# === Data Cleaning ===
def clean_data(data):
    required_columns = [
        'molregno', 'chembl_compound_id', 'canonical_smiles',
        'target_protein_accession', 'protein_sequence', 'ic50_nM'
    ]
    # drop rows where any of the listed columns is missing, and resets index numbering
    data = data.dropna(subset=required_columns).reset_index(drop=True)
    
    # Remove invalid SMILES
    data = data[data['canonical_smiles'].apply(lambda x: Chem.MolFromSmiles(x) is not None)]
    return data.reset_index(drop=True)


# === Batched Multi-GPU Inference ===
# get embeddings of drugs and proteins
def process_row_serial(df, save_path, batch_size=256):
    num_rows = len(df)
    all_drug_embs = []
    all_prot_embs = []
    all_ic50_vals = []

    total_batches = (num_rows + batch_size - 1) // batch_size

    for i, start in enumerate(range(0, num_rows, batch_size)):
        end = min(start + batch_size, num_rows)
        batch = df.iloc[start:end]

        print(f"Processing batch {i + 1} of {total_batches} ({start}–{end})")

        smiles_list = batch['canonical_smiles'].tolist()
        prot_seq_list = batch['protein_sequence'].tolist()
        ic50_vals = batch['ic50_nM'].tolist()

        # === Drug ===
        # can input list to tokenizer
        smiles_inputs = chembert_tokenizer(
            smiles_list, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        with torch.no_grad():
            drug_output = chembert_model(**smiles_inputs)
            drug_cls = drug_output.last_hidden_state[:, 0, :].cpu().numpy()

        # === Protein ===
        prot_seq_spaced = [' '.join(list(seq)) for seq in prot_seq_list]
        prot_inputs = protbert_tokenizer(
            prot_seq_spaced, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(device)

        with torch.no_grad():
            prot_output = protbert_model(**prot_inputs)
            prot_cls = prot_output.last_hidden_state[:, 0, :].cpu().numpy()

        all_drug_embs.extend(drug_cls.tolist())
        all_prot_embs.extend(prot_cls.tolist())
        all_ic50_vals.extend(ic50_vals)
        
        # memory use optimization - delete used variables
        del smiles_inputs, prot_inputs, drug_output, prot_output
        # garbage collection
        gc.collect()
        # empty cache
        torch.cuda.empty_cache()

    df_drug = pd.DataFrame(all_drug_embs).add_prefix("drug_")
    df_prot = pd.DataFrame(all_prot_embs).add_prefix("prot_")
    df_ic50 = pd.DataFrame({'ic50_nM': all_ic50_vals})

    final_df = pd.concat([df_drug, df_prot, df_ic50], axis=1)
    final_df.to_csv(save_path, index=False)


# append each 1/10 part of the whole data into one dataset
def append_dataframes():
    import pandas as pd
    
    # Step 1: Get list of your CSV files
    csv_files = [f"/scratch/eta/ml_training_data_part{i}.csv" for i in range(1, 11)]
    
    # Step 2: Read the first file (include header)
    df_list = []
    first_file = True
    
    for file in csv_files:
        if first_file:
            df = pd.read_csv(file)
            first_file = False
        else:
            df = pd.read_csv(file, header=0)  # Read but ignore the header
        df_list.append(df)
    
    # Step 3: Concatenate all DataFrames
    merged_df = pd.concat(df_list, ignore_index=True)
    
    # Step 4: Save to a new CSV with header
    merged_df.to_csv("/scratch/eta/ml_training_data_merged.csv", index=False)




# === Main ===
def main():
    
    input_path = '/scratch/eta/chembl_1st_data.csv'
    #embedding_path = '/scratch/eta/ml_training_data_1st.csv'
    #prediction_path = '/scratch/eta/model_predictions_1st.csv'

    data = pd.read_csv(input_path)
    df = clean_data(data)

    # Split into 10 parts
    df_parts = np.array_split(df, 10)
    
    # Process each part
    for i, part in enumerate(df_parts):
        print(f"Processing part {i+1}/5, rows {len(part)}")
        
        # Use a unique file per part
        part_path = f"/scratch/eta/ml_training_data_part{i+1}.csv"
        process_row_serial(part, part_path, batch_size=256)

    
    #process_row_serial(df, embedding_path, batch_size=256)
    
    append_dataframes()


if __name__ == "__main__":
    main()
