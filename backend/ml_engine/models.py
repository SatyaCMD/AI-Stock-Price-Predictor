from sklearn.linear_model import LinearRegression, LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input

def train_linear_regression(x_train, y_train):
    """
    Train a Linear Regression model.
    """
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

def train_logistic_regression(x_train, y_train):
    """
    Train a Logistic Regression model.
    """
    model = LogisticRegression()
    model.fit(x_train, y_train)
    return model

def build_lstm_model(input_shape):
    """
    Build an LSTM model with Dropout for better generalization.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(model, x_train, y_train, epochs=1, batch_size=1):
    """
    Train the LSTM model.
    """
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    return history
