#Importing python packages
import pandas as pd
import numpy as np
import pickle, os
from sklearn.preprocessing import RobustScaler, LabelEncoder

# Import the dataset
def import_dataset(path:str):
    dataset = pd.read_csv(path)
    dataset['timeslot'] = pd.to_datetime(dataset['timeslot'] , format = "%Y-%m-%d %H:%M:%S")
    dataset.ffill(inplace=True)
    return dataset

def import_test_dataset(path:str):
    dataset = pd.read_csv(path)
    dataset['timeslot'] = pd.to_datetime(dataset['timeslot'] , format = "%d-%m-%Y %H:%M")
    dataset.ffill(inplace=True)
    return dataset

# Feature engineering to compute impactul features
def compute_features(dataset):
    df = dataset.copy()
    df["diff"] = abs(df["max_metric_value"] - df["mean_metric_value"])
    df["ratio"] = df["max_metric_value"] / df["mean_metric_value"]
    df["delta_max"] = df["max_metric_value"].diff()
    df["delta_mean"] = df["mean_metric_value"].diff()

    # Z-score calculation with rolling window
    roll_w = 4
    rolling_mean_max = df["max_metric_value"].rolling(window=roll_w, min_periods=2).mean()
    rolling_std_max = df["max_metric_value"].rolling(window=roll_w, min_periods=2).std(ddof=0)
    rolling_std_max = rolling_std_max.replace(0, np.nan)
    df["zscore_max"] = (df["max_metric_value"] - rolling_mean_max) / rolling_std_max

    rolling_mean_mean = df["mean_metric_value"].rolling(window=roll_w, min_periods=2).mean()
    rolling_std_mean = df["mean_metric_value"].rolling(window=roll_w, min_periods=2).std(ddof=0)
    rolling_std_mean = rolling_std_mean.replace(0, np.nan)
    df["zscore_mean"] = (df["mean_metric_value"] - rolling_mean_mean) / rolling_std_mean
    return df

#Upsampling the data to increase the frequency to 1 minute
def upsample_per_interface(df):
    upsampled = []
    df['timeslot'] = pd.to_datetime(df['timeslot'])

    for name, group in df.groupby(['router_name']):
        group = group.sort_values('timeslot').copy()
        group = group.set_index('timeslot')

        # Resample to 1-minute frequency and interpolate numeric columns only
        group_resampled = group.resample('1T').interpolate(method='linear')
        
        # Broadcast router_name to all rows
        group_resampled['router'] = name[0]

        group_resampled.drop(columns=['router_name'], inplace=True)

        # Reset index
        upsampled.append(group_resampled.reset_index())

    return pd.concat(upsampled, ignore_index=True)

def impute_outliers_series(s: pd.Series) -> pd.Series:
    """Performing a process called winsorization to limit extreme values for model"""
    
    if s.dropna().empty:
        return s
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    IQR = q3 - q1
    upper_fence = q3 + 1.5 * IQR
    lower_fence = q1 - 1.5 * IQR

    # choose the max value that is <= upper_fence and min value that is >= lower_fence
    within_upper = s[~(s > upper_fence)]
    within_lower = s[~(s < lower_fence)]
    # fallback to series max/min if no values within fence
    upper = within_upper.max() if not within_upper.empty else s.max()
    lower = within_lower.min() if not within_lower.empty else s.min()

    res = s.copy()
    res.loc[s > upper] = upper
    res.loc[s < lower] = lower

    return res

# Impute outliers per group
def impute_outliers_group(g: pd.DataFrame) -> pd.DataFrame:

    cap_columns = ['max_metric_value', 'mean_metric_value', 'delta_max', 'delta_mean', 'diff', 'ratio', 'zscore_max', 'zscore_mean']
    
    g = g.copy()
    for col in cap_columns:
        if col in g.columns:
            g[col] = impute_outliers_series(g[col])

    return g

def encoding_router(final_dataset):
    # Encoding the categorical feature 'router' using LabelEncoder
    encoder = LabelEncoder()
    final_dataset['router_id'] = encoder.fit_transform(final_dataset['router'])
    with open(r'saved_binary_files\router_encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    return final_dataset

def building_lagged_sequences(final_dataset, window):

    routers = final_dataset['router_id'].unique()
    router_sequences = {}

    for r in routers:
        router_df = final_dataset[final_dataset['router_id'] == r].sort_values('timeslot')
        router_sequences[r] = router_df

    # Build sequences and split per-router (time-based) into train/val/test (80/10/10)
    window_size = window

    # prepare lists for train/val/test
    X_train, X_val, X_test = [], [], []
    y_train, y_val, y_test = [], [], []
    router_train, router_val, router_test = [], [], []

    numerical_features = ['max_metric_value', 'mean_metric_value', 'diff', 'ratio', 'delta_max', 'delta_mean', 'zscore_max', 'zscore_mean']

    for idx, r in enumerate(routers):
        df = router_sequences[r].sort_values('timeslot')
        data = df[numerical_features].values
        target = df[['max_metric_value','mean_metric_value']].values
        
        n = len(data)
        # number of usable samples for sequences
        num_samples = n - window_size
        if num_samples <= 0:
            continue

        # split counts based on samples (time-based order)
        train_count = int(num_samples * 0.8)
        val_count = int(num_samples * 0.1)
        # remaining goes to test
        test_count = num_samples - train_count - val_count
        # ensure non-negative
        if test_count < 0:
            test_count = 0

        for s in range(num_samples):
            seq = data[s:s+window_size]
            label = target[s+window_size]
            
            # assign by sample index (time-ordered)
            if s < train_count:
                X_train.append(seq); y_train.append(label); router_train.append(idx)
            elif s < train_count + val_count:
                X_val.append(seq); y_val.append(label); router_val.append(idx)
            else:
                X_test.append(seq); y_test.append(label); router_test.append(idx)

    # convert to arrays
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)
    router_train = np.array(router_train)
    router_val = np.array(router_val)
    router_test = np.array(router_test)

    print('Built sequences — train samples:', X_train.shape[0], ' val samples:', X_val.shape[0], ' test samples:', X_test.shape[0])

    return X_train, y_train, router_train, X_val, y_val, router_val, X_test, y_test, router_test, routers

def scale_data(X_train, y_train, router_train, X_val, y_val, router_val,  X_test, y_test, router_test, routers):
    # Store scaled sequences
    X_train_scaled, X_val_scaled = [], []
    Y_train_scaled, Y_val_scaled= [], []

    # Store router-wise scalers (optional for later inverse transform)
    router_train_scalers = {}
    router_test_scalers = {}

    for router_idx, r in enumerate(routers):

        # Mask samples belonging to this router
        train_mask = router_train == router_idx
        val_mask = router_val == router_idx
        test_mask = router_test == router_idx

        # Extract router-specific training samples
        X_train_router = X_train[train_mask]  # shape: (samples, window_size, num_features)
        X_val_router = X_val[val_mask]
        X_test_router = X_test[test_mask]

        y_train_router = y_train[train_mask]  # shape: (samples, window_size, num_features)
        y_val_router = y_val[val_mask]
        y_test_router = y_test[test_mask]

        # Reshape to 2D for scaling: (samples * timesteps, num_features)
        X_train_flat = X_train_router.reshape(-1, X_train_router.shape[-1])
        Y_train_flat = y_train[train_mask].reshape(-1, y_train.shape[-1])

        # Fit scaler only on training data for this router
        scaler_train = RobustScaler()
        scaler_train.fit(X_train_flat)

        scaler_test = RobustScaler()
        scaler_test.fit(Y_train_flat)

        # Transform train/val/test
        X_train_scaled_router = scaler_train.transform(X_train_router.reshape(-1, X_train_router.shape[-1])).reshape(X_train_router.shape)
        X_val_scaled_router = scaler_train.transform(X_val_router.reshape(-1, X_val_router.shape[-1])).reshape(X_val_router.shape)

        # Transform targets
        Y_train_scaled_router = scaler_test.transform(y_train_router.reshape(-1, y_train_router.shape[-1])).reshape(y_train_router.shape)
        Y_val_scaled_router = scaler_test.transform(y_val_router.reshape(-1, y_val_router.shape[-1])).reshape(y_val_router.shape)

        # Append scaled data (in the same order as before)
        X_train_scaled.append(X_train_scaled_router)
        X_val_scaled.append(X_val_scaled_router)

        # Append scaled data (in the same order as before)
        Y_train_scaled.append(Y_train_scaled_router)
        Y_val_scaled.append(Y_val_scaled_router)

        # Keep track of scalers
        router_train_scalers[r] = scaler_train
        router_test_scalers[r] = scaler_test

    # Concatenate all routers back
    X_train_scaled = np.concatenate(X_train_scaled, axis=0)
    X_val_scaled = np.concatenate(X_val_scaled, axis=0)

    Y_train_scaled = np.concatenate(Y_train_scaled, axis=0)
    Y_val_scaled = np.concatenate(Y_val_scaled, axis=0)

    with open(r'saved_binary_files\router_scalers_train.pkl', 'wb') as f:
        pickle.dump(router_train_scalers, f)

    with open(r'saved_binary_files\router_scalers_test.pkl', 'wb') as g:
        pickle.dump(router_test_scalers, g)

    return X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled

def test_encoder(final_dataset):

    with open(r'd:\AI-ES\Projects\KPN Anamoly case\AIEngineerAssignment\scalers\router_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)

    final_dataset['router_id'] = encoder.transform(final_dataset['router'])

    return final_dataset

def create_lagged_test_sequences(final_dataset, window):

    routers = final_dataset['router_id'].unique()
    router_sequences = {}

    for r in routers:
        router_df = final_dataset[final_dataset['router_id'] == r].sort_values('timeslot')
        router_sequences[r] = router_df

    window_size = window

    X_pred, Y_pred, router_pred = [], [], []

    numerical_features = ['max_metric_value', 'mean_metric_value', 'diff', 'ratio', 'delta_max', 'delta_mean', 'zscore_max', 'zscore_mean']

    for idx, r in enumerate(routers):
        df = router_sequences[r].sort_values('timeslot')
        data = df[numerical_features].values
        target = df[['max_metric_value','mean_metric_value']].values

        n = len(data)
        # number of usable samples for sequences
        num_samples = n - window_size

        
        for s in range(num_samples):
            seq = data[s:s+window_size]
            label = target[s+window_size]

            X_pred.append(seq); Y_pred.append(label); router_pred.append(idx)

    X_pred = np.array(X_pred)
    Y_pred = np.array(Y_pred)
    router_pred = np.array(router_pred)

    print('Built sequences — train samples:', X_pred.shape[0], ' val samples:', Y_pred.shape[0], ' test samples:', router_pred.shape[0])

    return X_pred, Y_pred, router_pred, routers

def scale_test_data(X_pred, Y_pred, router_pred, routers):
    
    with open(r'd:\AI-ES\Projects\KPN Anamoly case\AIEngineerAssignment\scalers\router_scalers_train.pkl', 'rb') as f:
        router_train_scalers = pickle.load(f)

    # Load test scalers
    with open(r'd:\AI-ES\Projects\KPN Anamoly case\AIEngineerAssignment\scalers\router_scalers_test.pkl', 'rb') as g:
        router_test_scalers = pickle.load(g)

    X_pred_scaled_all , Y_pred_scaled_all = [], []

    for router_idx, r in enumerate(routers):

        X_mask = router_pred == router_idx
        Y_mask = router_pred == router_idx

        X_data = X_pred[X_mask]  # shape: (samples, window_size, num_features)
        Y_data = Y_pred[Y_mask]
        
        X_scaler = router_train_scalers[r]
        Y_scaler = router_test_scalers[r]

        X_pred_scaled = X_scaler.transform(X_data.reshape(-1, X_data.shape[-1])).reshape(X_data.shape)
        Y_pred_scaled = Y_scaler.transform(Y_data.reshape(-1, Y_data.shape[-1])).reshape(Y_data.shape)

        X_pred_scaled_all.append(X_pred_scaled)
        Y_pred_scaled_all.append(Y_pred_scaled)

    X_pred_scaled_all = np.concatenate(X_pred_scaled_all, axis=0)
    Y_pred_scaled_all = np.concatenate(Y_pred_scaled_all, axis=0)

    return X_pred_scaled_all, Y_pred_scaled_all