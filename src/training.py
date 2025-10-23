#Import necessary libraries
import pandas as pd
import numpy as np
import pickle, os
from sklearn.preprocessing import RobustScaler, LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Concatenate, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from data_preparation import import_dataset, compute_features, upsample_per_interface, impute_outliers_series, impute_outliers_group, encoding_router, building_lagged_sequences, scale_data
from model import build_model
from resources.config import training_path, lag
from logs.log_setup import get_logger

# Set random seed for reproducibility
np.random.seed(42)

def main():

    print("Starting the training process")

    logger = get_logger("Training Cycle")
    logger.info("Starting Training cycle")

    try:
        dataset = import_dataset(training_path)
    except Exception as e:
        logger.error(f"Error importing dataset: {e}")
        return None

    print("Dataset imported successfully")
    logger.info("Successfully imported the dataset")

    # Combine hostname and ifname into a single identifier
    dataset["router_name"] = dataset["hostname"] + "-" + dataset["ifname"]
    dataset.drop(columns=["hostname", "ifname"], inplace=True)

    #Upsampling into higher frequency (1 minute)
    dataset_upsampled = upsample_per_interface(dataset)
    logger.info("Upsampling completed successfully")

    print("Upsampling completed")

    dataset_upsampled.sort_values(['router', 'timeslot'], inplace=True)

    # Compute feature values per router
    final_dataset = dataset_upsampled.groupby('router', group_keys=False).apply(compute_features).reset_index(drop=True)
    logger.info("Computed additional features successfully")
    final_dataset.dropna(subset=['delta_max','delta_mean','ratio','zscore_max'], how='any', inplace=True)

    final_dataset = final_dataset.groupby('router', group_keys=False).apply(impute_outliers_group).reset_index(drop=True)

    print("Feature engineering and outlier imputation completed")

    final_dataset = encoding_router(final_dataset)
    logger.info("Encoded the router IDs successfully")

    X_train, y_train, router_train, X_val, y_val, router_val,  X_test, y_test, router_test, routers = building_lagged_sequences(final_dataset, window = lag)

    logger.info("Created lagged sequences for training, validation, and testing")

    X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled = scale_data(X_train, y_train, router_train, X_val, y_val, router_val,  X_test, y_test, router_test, routers)

    logger.info("Scaled the data for training, validation, and testing")

    print("Data preparation completed, starting model training")

    model = build_model(routers, numerical_features=['max_metric_value', 'mean_metric_value', 'diff', 'ratio', 'delta_max', 'delta_mean', 'zscore_max', 'zscore_mean'], window_size=lag)

    logger.info("Model architecture created successfully")

    history = model.fit(
        [X_train_scaled, router_train],
        Y_train_scaled,
        validation_data=([X_val_scaled, router_val], Y_val_scaled),
        epochs=20,
        batch_size=32,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
            ModelCheckpoint(r'saved_binary_files\best_model.h5', save_best_only=True, monitor='val_loss')
        ]
    )
    
    logger.info("Model training completed successfully")

if __name__ == "__main__":
    main()