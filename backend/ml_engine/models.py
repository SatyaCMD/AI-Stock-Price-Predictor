from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def train_linear_regression(x_train, y_train):
    """
    Train a Linear Regression model.
    """
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

def build_lstm_model(input_shape):
    """
    Build an LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(model, x_train, y_train, epochs=1, batch_size=1):
    """
    Train the LSTM model.
    """
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    return model
