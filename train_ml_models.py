#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 14:58:41 2025

@author: eta

Train ML models and get error metrics + pearson coefficient values

Run on 1 cpu
"""
import pandas as pd
import numpy as np
import time

# === Model Training ===
def train_ml_model(test_size=0.2, random_state=42, n_jobs=1):
    from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import SGDRegressor
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # load data
    df = pd.read_csv('/scratch/eta/ml_training_data_ae_reduced.csv')
    
    # zero and negative IC50 values are invalid and should generally be considered errors or artifacts. IC50 is a concentration value (usually in nanomolar), so it must be strictly positive:
    # Zero → implies infinite potency — biologically meaningless.
    # Negative → impossible, likely a data entry error or a result of processing mistake.
    df = df[df['ic50_nM'] > 0].reset_index(drop=True)
    
    # Convert nM to M (1 nM = 1e-9 M)
    # use pIC50, which reduces the data variability
    df['pIC50'] = -np.log10(df['ic50_nM'] * 1e-9)

    X = df.drop(columns=['ic50_nM', 'pIC50'])
    y = df['pIC50']

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features (fit on train, transform all)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = []
    #models.append(('LR', LinearRegression()))
    #models.append(('KNN', KNeighborsRegressor()))
    #models.append(('SVM', SVR()))
    #models.append(('MLP', MLPRegressor()))
    #models.append(('DT', DecisionTreeRegressor()))
    #models.append(('RF', RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=n_jobs)))
    #models.append(('AB', AdaBoostRegressor()))
    #models.append(('GB', GradientBoostingRegressor()))
    #models.append(('SGD', SGDRegressor()))
    
    model_names = ['LR', 'KNN', 'SVM', 'MLP', 'DT', 'RF', 'AB', 'GB', 'SGD']
    
    # use only default parameters of each model
    models.append(('LR', LinearRegression(
        fit_intercept=True,
        #normalize='deprecated',  # deprecated since v1.0, ignored if fit_intercept=False
        copy_X=True,
        n_jobs=None,
        positive=False
    )))
    
    models.append(('KNN', KNeighborsRegressor(
        n_neighbors=5,              # default
        weights='uniform',          # default
        algorithm='auto',           # default
        leaf_size=30,                # default
        p=2,                         # default (Euclidean)
        metric='minkowski',          # default
        n_jobs=n_jobs
    )))
    
    models.append(('SVM', SVR(
        kernel='rbf',                # default
        degree=3,                    # default
        gamma='scale',               # default
        coef0=0.0,                    # default
        tol=1e-3,                     # default
        C=1.0,                        # default
        epsilon=0.1,                  # default
        shrinking=True,               # default
        cache_size=200,               # default
        max_iter=-1                   # default
    )))
    
    models.append(('MLP', MLPRegressor(
        hidden_layer_sizes=(100,),   # default
        activation='relu',           # default
        solver='adam',               # default
        alpha=0.0001,                 # default
        batch_size='auto',            # default
        learning_rate='constant',     # default
        learning_rate_init=0.001,     # default
        power_t=0.5,                  # default
        max_iter=200,                 # default
        shuffle=True,                 # default
        random_state=random_state,
        tol=1e-4,                     # default
        momentum=0.9,                 # default
        nesterovs_momentum=True,      # default
        early_stopping=False,         # default
        validation_fraction=0.1,      # default
        beta_1=0.9,                   # default
        beta_2=0.999,                 # default
        epsilon=1e-8,                 # default
        n_iter_no_change=10,          # default
        max_fun=15000                 # default
    )))
    
    models.append(('DT', DecisionTreeRegressor(
        criterion='squared_error',   # default
        splitter='best',              # default
        max_depth=None,               # default
        min_samples_split=2,          # default
        min_samples_leaf=1,           # default
        min_weight_fraction_leaf=0.0, # default
        max_features=None,            # default
        random_state=random_state,
        max_leaf_nodes=None,          # default
        min_impurity_decrease=0.0,    # default
        ccp_alpha=0.0                 # default
    )))
    
    models.append(('RF', RandomForestRegressor(
        n_estimators=100,             # default
        criterion='squared_error',    # default
        max_depth=None,               # default
        min_samples_split=2,          # default
        min_samples_leaf=1,           # default
        min_weight_fraction_leaf=0.0, # default
        max_features=1.0,             # default
        max_leaf_nodes=None,          # default
        min_impurity_decrease=0.0,    # default
        bootstrap=True,               # default
        oob_score=False,               # default
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=0,                    # default
        warm_start=False,              # default
        ccp_alpha=0.0,                 # default
        max_samples=None               # default
    )))
    
    models.append(('AB', AdaBoostRegressor(
        estimator=None,                # default
        n_estimators=50,                # default
        learning_rate=1.0,              # default
        loss='linear',                  # default
        random_state=random_state
    )))
    
    models.append(('GB', GradientBoostingRegressor(
        loss='squared_error',          # default
        learning_rate=0.1,             # default
        n_estimators=100,              # default
        subsample=1.0,                  # default
        criterion='friedman_mse',      # default
        min_samples_split=2,           # default
        min_samples_leaf=1,            # default
        min_weight_fraction_leaf=0.0,  # default
        max_depth=3,                   # default
        min_impurity_decrease=0.0,     # default
        init=None,                     # default
        random_state=random_state,
        max_features=None,             # default
        alpha=0.9,                     # default
        verbose=0,                     # default
        max_leaf_nodes=None,           # default
        warm_start=False,              # default
        validation_fraction=0.1,       # default
        n_iter_no_change=None,         # default
        tol=1e-4,                      # default
        ccp_alpha=0.0                  # default
    )))
    
    models.append(('SGD', SGDRegressor(
        loss='squared_error',          # default
        penalty='l2',                  # default
        alpha=0.0001,                   # default
        l1_ratio=0.15,                  # default
        fit_intercept=True,             # default
        max_iter=1000,                  # default
        tol=1e-3,                       # default
        shuffle=True,                   # default
        verbose=0,                      # default
        epsilon=0.1,                    # default
        random_state=random_state,
        learning_rate='invscaling',     # default
        eta0=0.01,                      # default
        power_t=0.25,                   # default
        early_stopping=False,           # default
        validation_fraction=0.1,        # default
        n_iter_no_change=5,             # default
        warm_start=False,               # default
        average=False                   # default
    )))

    # lists for storing error metrics and pearson coefficients
    rmse_list = []
    r2_list = []
    mae_list = []
    pearson_coef_list = []
    
    # train each model, evaluate on test set, get performance results (save as csv, plot as barplot)
    for name, model in models:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        rmse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        #pearson_coef = y_test.corr(y_pred, method='pearson')
        pearson_coef = y_test.corr(pd.Series(y_pred, index=y_test.index), method='pearson')
        print('model training for: ', name)
        print('RMSE, R2, MAE, PEARSON-COEF: ', rmse, r2, mae, pearson_coef)
        rmse_list.append(rmse)
        r2_list.append(r2)
        mae_list.append(mae)
        pearson_coef_list.append(pearson_coef)
        pd.DataFrame({
            'true_ic50_nM': y_test.values,
            'predicted_ic50_nM': y_pred
        }).to_csv(f'/scratch/eta/predictions_{name}_2nd_ae.csv', index=False)
        print(f"Saved test predictions to /scratch/eta/predictions_{name}_2nd_ae.csv")
    
    # make barplot - compare algorithms, model selection
    error_df = pd.DataFrame({
        'Model': model_names,
        'RMSE': rmse_list,
        'R2': r2_list,
        'MAE': mae_list
        })
    error_df.to_csv('/scratch/eta/errors_models_2nd_ae.csv')
    # Melt to long form for grouped barplot
    error_df_melted = error_df.melt(id_vars='Model', 
                                    var_name='Metric', 
                                    value_name='Value')
    
    # Plot grouped bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(data=error_df_melted, x='Model', y='Value', hue='Metric')
    plt.title('Performance Error Metrics for Each Model')
    plt.xlabel('Models')
    plt.ylabel('Metric Value')
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig('/sybig/home/eta/Internship_2025_sysbio/errors_models_2nd_ae.png', dpi=300, bbox_inches="tight")
    
    # plot pearson coefficient bar plot
    pearson_coef_df = pd.DataFrame({
        'Model': model_names,
        'Pearson Coefficient': pearson_coef_list
        })
    pearson_coef_df.to_csv('/scratch/eta/pearson_coefs_models_2nd_ae.csv')
    plt.figure(figsize=(10, 6))
    plt.bar(pearson_coef_df['Model'], pearson_coef_df['Pearson Coefficient'], color='skyblue')
    plt.title("Pearson's correlation coefficients for each model")
    plt.xlabel('Models')
    plt.ylabel("Pearson's correlation coefficient")
    plt.tight_layout()
    plt.savefig('/sybig/home/eta/Internship_2025_sysbio/pearson_coefs_models_2nd_ae.png', dpi=300, bbox_inches="tight")
    


def main():
    '''
    # 1. Check target distribution
    import matplotlib.pyplot as plt
    df = pd.read_csv('/scratch/eta/ml_training_data_pca_reduced.csv')
    #plt.hist(df['ic50_nM'], bins=1000)
    #plt.yscale('log')
    #plt.title('Raw IC50 Distribution')
    #plt.savefig('/scratch/eta/ml_training_data_ic50_distribution_2.png', dpi=300, bbox_inches="tight")
    
    import numpy as np
    # Filter out zero or negative IC50 values
    valid_ic50 = df['ic50_nM'][df['ic50_nM'] > 0]
    
    # Apply log10
    log_ic50 = np.log10(valid_ic50)
    plt.hist(log_ic50, bins=1000)
    plt.title('Log10 IC50 Distribution')
    plt.xlabel('log10(IC50_nM)')
    plt.ylabel('Frequency')
    plt.savefig('/scratch/eta/ml_training_data_ic50_distribution_3.png', dpi=300, bbox_inches="tight")
    
    num_zero = (df['ic50_nM'] == 0).sum()
    num_negative = (df['ic50_nM'] < 0).sum()
    
    print(f"Number of IC50 values equal to 0: {num_zero}")
    print(f"Number of IC50 values less than 0: {num_negative}")
    
    # Number of IC50 values equal to 0: 547
    # Number of IC50 values less than 0: 19
    '''    
    
    start_time = time.time()
    #train_ml_model('/scratch/eta/ml_training_data_pca_reduced.csv', '/scratch/eta/random_forest_predictions_test2.csv')
    train_ml_model()
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"--- {int(hours)}h {int(minutes)}m {seconds:.2f}s ---")
    #print("--- %s seconds ---" % (end_time - start_time))
    
if __name__ == "__main__":
    main()