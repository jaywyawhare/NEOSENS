import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Reshape

def createModel(model_type, num_layers, layer_types, layer_units, input_shape, num_output_units, loss_function, optimizer, metrics):
    '''Create a deep learning model with specified architecture.

    Args:
        model_type (str): The type of model to create. Supported values are
            'Sequentials', 'LSTM', 'Convolutional'.
        num_layers (int): The number of layers to include in the model.
        layer_types (list[str]): A list of layer types to include in the model.
        layer_units (list[int]): A list of the number of units for each layer in the model.
        input_shape (tuple[int]): The shape of the input data.
        num_output_units (int): The number of output units in the final layer of the model.
        loss_function (str): The loss function to use for training.
        optimizer (str): The optimizer to use for training.
        metrics (list[str]): A list of metrics to monitor during training.

    Returns:
        tensorflow.keras.models.Model: The compiled deep learning model.

    Raises:
        ValueError: If an unsupported model type, loss function, optimizer, or metric is specified.
    '''

    model = Sequential()
    if model_type == 'Sequentials':
        for i in range(num_layers):
            if layer_types[i] == 'Dense':
                model.add(Dense(layer_units[i], activation='relu'))
            elif layer_types[i] == 'Dropout':
                model.add(Dropout(layer_units[i]))
            else:
                raise ValueError(f"Unsupported layer type: {layer_types[i]}")
    
    elif model_type == 'LSTM':
        model.add(Reshape((input_shape[1], 1)))
        for i in range(num_layers):
            if layer_types[i] == 'LSTM':
                model.add(LSTM(layer_units[i], input_shape=input_shape))
            elif layer_types[i] == 'Dense':
                model.add(Dense(layer_units[i], activation='relu'))
            elif layer_types[i] == 'Dropout':
                model.add(Dropout(layer_units[i]))
            else:
                raise ValueError(f"Unsupported layer type: {layer_types[i]}")

    elif model_type == 'Convolutional1D':
        model.add(Reshape((input_shape[1], 1)))
        for i in range(num_layers):
            if layer_types[i] == 'Conv1D':
                model.add(Conv1D(filters=layer_units[i], kernel_size=3, activation='relu', input_shape=input_shape))
            elif layer_types[i] == 'MaxPooling1D':
                model.add(MaxPooling1D(pool_size=2))
            elif layer_types[i] == 'Conv2D':
                model.add(Conv2D(filters=layer_units[i], kernel_size=3, activation='relu', input_shape=input_shape))
            elif layer_types[i] == 'MaxPooling2D':
                model.add(MaxPooling2D(pool_size=2))
            elif layer_types[i] == 'Flatten':
                model.add(Flatten())
            elif layer_types[i] == 'Dense':
                model.add(Dense(layer_units[i], activation='relu'))
            elif layer_types[i] == 'Dropout':
                model.add(Dropout(layer_units[i]))
            else:
                raise ValueError(f"Unsupported layer type: {layer_types[i]}")

    elif model_type == 'Convolutional2D':
        model.add(Reshape((input_shape[1], 1)))
        for i in range(num_layers):
            if layer_types[i] == 'Conv2D':
                model.add(Conv1D(filters=layer_units[i], kernel_size=3, activation='relu', input_shape=input_shape))
            elif layer_types[i] == 'MaxPooling2D':
                model.add(MaxPooling1D(pool_size=2))
            elif layer_types[i] == 'Flatten':
                model.add(Flatten())
            elif layer_types[i] == 'Dense':
                model.add(Dense(layer_units[i], activation='relu'))
            elif layer_types[i] == 'Dropout':
                model.add(Dropout(layer_units[i]))
            else:
                raise ValueError(f"Unsupported layer type: {layer_types[i]}")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.add(Dense(num_output_units, activation='softmax'))
    model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
    model.build(input_shape=input_shape)
    model.summary()
    return model
