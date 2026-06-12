# Drug Protein Interactions



## 1. chembl_access.py
Get data from ChEMBL

## 2. get_embeddings.py
Get embeddings of ChemBERTa for drug SMILES strings and ProtBERT for protein amino acid sequences

## 3. pca_dimension_reduction.py
Reduce dimensions of the data using PCA

## 4. autoencoder_dimension_reduction.py
Reduce dimensions of the data using Autoencoders

## 5. train_ml_models.py
Train several Machine Learning models for the two reduced data, get performance metrics

## Dockerfile, requirements.txt
Files used to generate docker image