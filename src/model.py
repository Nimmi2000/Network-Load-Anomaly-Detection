from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Concatenate, Flatten, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from config import embedding_layer, lstm_units, dense_units, dense_activation, dropout_rate, optimizer_train, learning_rate

def get_optimizer(name: str, lr: float):
    if name.lower() == "adam":
        return Adam(learning_rate=lr)
    elif name.lower() == "rmsprop":
        return RMSprop(learning_rate=lr)
    elif name.lower() == "sgd":
        return SGD(learning_rate=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

def build_model(routers, numerical_features, window_size):

    n_routers = len(routers)
    embedding_dimension = embedding_layer

    # Input layers
    seq_input = Input(shape=(window_size, len(numerical_features)))
    router_input = Input(shape=(1,))

    # Embedding layer for router
    router_embedding = Embedding(input_dim=n_routers, output_dim=embedding_dimension)(router_input)
    router_embedding = Flatten()(router_embedding)

    # LSTM first layer
    x1 = LSTM(lstm_units, return_sequences=False)(seq_input)

    # Combining Embedding output with LSTM output
    x_combined = Concatenate()([x1, router_embedding])

    # Output layer
    x = Dense(dense_units, activation=dense_activation, kernel_regularizer=regularizers.l2(1e-4))(x_combined)
    x = Dropout(dropout_rate)(x)
    output = Dense(2)(x)

    model = Model(inputs=[seq_input, router_input], outputs=output)
    optimizer = get_optimizer(optimizer_train, learning_rate)

    model.compile(optimizer=optimizer, loss='mse')

    return model