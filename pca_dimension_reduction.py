#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 12:27:59 2025

@author: eta

pca dimension reduction - run on cpu (parallel)
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import pickle
from joblib import load

CHEM_PCA_MODEL_PATH = "/scratch/eta/pca_model_chemberta.pkl"
PROT_PCA_MODEL_PATH = "/scratch/eta/pca_model_protbert.pkl"
n_samples_for_pca = 100
pca_n_components = 50

# make pca model of chemberta drug embeddings, using first 100 rows as sample and 50 pca components, save pca model as pickle
def pca_model_chemberta(dataset_path):
    df = pd.read_csv(dataset_path)
    sample = df.iloc[:n_samples_for_pca, 0:768]   # the first 768 columns are the drug cls tokens (768-dimensional vector)
    # change sample into array of vectors (each row)
    vectors = sample.to_numpy()

    print(f"Vector array shape: {vectors.shape}")
    print("Any NaNs in vectors?", np.isnan(vectors).any())
    print("Any Infs in vectors?", np.isinf(vectors).any())

    pca_model = PCA(n_components=pca_n_components)
    pca_model.fit(vectors)

    with open(CHEM_PCA_MODEL_PATH, "wb") as f:
        pickle.dump(pca_model, f)
    print('PCA model (drug) saved.')
    
    #return pca_model

# make pca model of protbert protein embeddings, using first 100 rows as sample and 50 pca components, save pca model as pickle
def pca_model_protbert(dataset_path):
    df = pd.read_csv(dataset_path)
    sample = df.iloc[:n_samples_for_pca, 768:1792]    # 768:(768+1024) - these are columns containing protein cls tokens (1024-dimensional vector)
    # change sample into array of vectors (each row)
    vectors = sample.to_numpy()

    print(f"Vector array shape: {vectors.shape}")
    print("Any NaNs in vectors?", np.isnan(vectors).any())
    print("Any Infs in vectors?", np.isinf(vectors).any())

    pca_model = PCA(n_components=pca_n_components)
    pca_model.fit(vectors)

    with open(PROT_PCA_MODEL_PATH, "wb") as f:
        pickle.dump(pca_model, f)
    print('PCA model (protein) saved.')
    
    #return pca_model

# load the models globally to reduce memory overhead
pca_chemberta = load(CHEM_PCA_MODEL_PATH)
pca_protbert = load(PROT_PCA_MODEL_PATH)

'''
def pca_reduce_chemberta(cls_embeddings):  # input one line, 768 dim vector
    #with open(CHEM_PCA_MODEL_PATH, "rb") as f:
    #    pca = pickle.load(f)
    pca_embs = pca_chemberta.transform(cls_embeddings)
    pca_flat = pca_embs.flatten()  # pca_flat is 1D numpy array
    return pca_flat
    


def pca_reduce_protbert(cls_embeddings):
    #with open(PROT_PCA_MODEL_PATH, "rb") as f:  # input one line, 1024 dim vector
    #    pca = pickle.load(f)
    pca_embs = pca_protbert.transform(cls_embeddings)
    pca_flat = pca_embs.flatten()  # pca_flat is 1D numpy array
    return pca_flat
'''  
    
# for each row, transform the chemberta embeddings using pca model of chemberta, and the protbert embeddings using pca model of protbert, and return the transformed row, together with target column ic50
def reduce_row(row):
    chemberta_vec = pca_chemberta.transform([row[0:768].to_numpy()])[0]
    protbert_vec = pca_protbert.transform([row[768:1792].to_numpy()])[0]
    ic50 = row['ic50_nM']
    return np.concatenate([chemberta_vec, protbert_vec, [ic50]])
    
# parallelize the previous function, using 20 cores, save results as csv file
def reduce_row_parallel(data_path, save_path):
    import pandas as pd
    from joblib import Parallel, delayed
    
    # df shape: (N, 1792 + 1) → 768 chemberta + 1024 protbert + ic50
    df = pd.read_csv(data_path)
    
    # Apply row-wise in parallel
    results = Parallel(n_jobs=20)(
        delayed(reduce_row)(row) for _, row in df.iterrows()
    )
    
    # Convert results to DataFrame
    n_chemberta_pca = pca_n_components
    n_protbert_pca = pca_n_components
    
    columns = (
        [f"chemberta_pca_{i}" for i in range(n_chemberta_pca)] +
        [f"protbert_pca_{i}" for i in range(n_protbert_pca)] +
        ['ic50_nM']
    )
    
    df_reduced = pd.DataFrame(results, columns=columns)
    df_reduced.to_csv(save_path, index=False)    
    
    
    
    
    
def main():
    #pca_model_chemberta('/scratch/eta/ml_training_data_merged.csv')  # merged csv
    #pca_model_protbert('/scratch/eta/ml_training_data_merged.csv')   # merged csv
    
    reduce_row_parallel('/scratch/eta/ml_training_data_merged.csv', '/scratch/eta/ml_training_data_pca_reduced.csv')
    
    
    
if __name__ == "__main__":
    main()