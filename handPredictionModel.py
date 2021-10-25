import tensorflow as tf
import time
import joblib

class HandPredictionModel(tf.keras.Model):

    def __init__(self):
        super().__init__(self)
        self.lstm1 = tf.keras.layers.LSTM(512, return_sequences=True, return_state=True)
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.lstm2 = tf.keras.layers.LSTM(512, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(156, activation="sigmoid")
            
    @tf.function
    def call(self, x, lstm_1_states=None, lstm_2_states=None, return_state=False, training=False):
        if lstm_1_states is None:
            lstm_1_states = self.lstm1.get_initial_state(x)
        if lstm_2_states is None:
            lstm_2_states = self.lstm2.get_initial_state(x)
        x, h1, c1 = self.lstm1(x, lstm_1_states)
        x = self.dropout1(x)
        x, h2, c2 = self.lstm2(x, lstm_2_states)
        x = self.dense(x)
        if return_state:
            return x, c1, h1, c2, h2
        return x
