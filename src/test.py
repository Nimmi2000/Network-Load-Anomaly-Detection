import pandas as pd
import numpy as np
import pickle, os
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import argparse
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Concatenate, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model
from logs.log_setup import get_logger

from data_preparation import import_test_dataset, compute_features, test_encoder, create_lagged_test_sequences, scale_test_data
from resources.config import test_path, lag

def main():

    parser = argparse.ArgumentParser(description="Train an LSTM model")

    # define arguments
    parser.add_argument("--routernumber", type=int, default=0, help="Enter the router number to predict")

    args = parser.parse_args()

    logger = get_logger("Inference Cycle")
    logger.info("Starting inference cycle")

    try:
        dataset = import_test_dataset(test_path)
    except Exception as e:
        logger.error(f"Error importing dataset: {e}")
        return None
    
    logger.info("Imported Dataset")

    dataset.sort_values(['router', 'timeslot'], inplace=True)

    final_dataset = dataset.groupby('router', group_keys=False).apply(compute_features).reset_index(drop=True)
    logger.info("Computed Additional Features")
    final_dataset.dropna(subset=['delta_max','delta_mean','ratio','zscore_max'], how='any', inplace=True)

    final_dataset = test_encoder(final_dataset)
    logger.info("Encoded the router IDs")

    X_pred, Y_pred, router_pred, routers = create_lagged_test_sequences(final_dataset, window = lag)
    logger.info("Created lagged sequences for Inference")

    X_pred_scaled, Y_pred_scaled = scale_test_data(X_pred, Y_pred, router_pred, routers)
    logger.info("Scaled the test data for Inference")

    try:
        model = load_model(r'saved_binary_files\best_model.h5', compile=False)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None
    logger.info("Trained model loaded successfully")

    router_to_predict = args.routernumber
    # Indices where router_test == 0
    indices = [i for i, value in enumerate(router_pred) if value == router_to_predict]

    # Select rows
    X_selected = X_pred_scaled[indices]
    Y_selected = Y_pred_scaled[indices]
    router_selected = router_pred[indices]

    predictions = model.predict([X_selected, router_selected])
    logger.info("The results have been predicted by LSTM")

    time = np.arange(len(X_selected))

    plt.figure(figsize=(14, 6))

    # Plot Feature 1
    plt.subplot(2, 1, 1)
    plt.plot(time, Y_selected[:, 0], label='Actual Feature 1', linewidth=1)
    plt.plot(time, predictions[:, 0], label='Predicted Feature 1', linewidth=1, alpha=0.7)
    plt.title('Feature 1: Actual vs Predicted')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    # Plot Feature 2
    plt.subplot(2, 1, 2)
    plt.plot(time, Y_selected[:, 1], label='Actual Feature 2', linewidth=1)
    plt.plot(time, predictions[:, 1], label='Predicted Feature 2', linewidth=1, alpha=0.7)
    plt.title('Feature 2: Actual vs Predicted')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    error = abs(predictions - Y_selected)
    time = final_dataset[final_dataset['router_id'] == router_to_predict]['timeslot'][6:]
    max = final_dataset[final_dataset['router_id'] == router_to_predict]['max_metric_value'][6:]
    mean = final_dataset[final_dataset['router_id'] == router_to_predict]['mean_metric_value'][6:]

    logger.info("Startig DBSCAN clustering for anomaly detection")
    dbscan = DBSCAN(eps=1, min_samples=150)  # Increase eps
    labels = dbscan.fit_predict(error)

    logger.info("DBSCAN clustering completed")

    anomalies = error[labels == -1]

    if len(anomalies) == 0:
        logger.warning("No anomalies detected by DBSCAN. Checking for mean error.")
        # Compute cluster center (mean of all points)
        center = np.mean(error)
        
        # Step 4: Define a threshold for “too far from zero”
        center_threshold = 4.0  # <-- tune this for your data
        
        # Step 5: If the mean error is too far from zero → mark all as anomalies
        if np.abs(center) > center_threshold:
            labels = np.array([-1]*len(error))  # Mark all as anomalies

    plotting_dataset = pd.DataFrame({'time': time, "max_metric_value": max, "mean_metric_value": mean, "label": labels})

        # Ensure time is a datetime
    plotting_dataset["time"] = pd.to_datetime(plotting_dataset["time"])

    # Sort by time just in case
    plotting_dataset = plotting_dataset.sort_values("time")

    # --- Plot setup ---
    plt.figure(figsize=(14, 6))

    # Plot the two metric lines
    plt.plot(plotting_dataset["time"], plotting_dataset["max_metric_value"], label="Max Metric", linewidth=1.5)
    plt.plot(plotting_dataset["time"], plotting_dataset["mean_metric_value"], label="Mean Metric", linewidth=1.5)

    # --- Highlight anomaly periods ---
    # Assuming label == -1 means anomaly
    anomaly_mask = plotting_dataset["label"] == -1
    anomaly_times = plotting_dataset.loc[anomaly_mask, "time"]

    # Marking anomaly regions
    plt.fill_between(plotting_dataset["time"], 
                    plotting_dataset["max_metric_value"].min(), 
                    plotting_dataset["max_metric_value"].max(),
                    where=anomaly_mask,
                    color="red", alpha=0.1, label="Anomaly region")

    # Plotting values

    plt.title("Metric Trends with Anomaly Indication", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()