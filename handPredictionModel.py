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
        # if lstm_1_states is None:
        #     lstm_1_states = self.lstm1.get_initial_state(x)
        # if lstm_2_states is None:
        #     lstm_2_states = self.lstm2.get_initial_state(x)
        # # print("LSTM 1 STATES:")
        # # print(lstm_1_states)
        try:
            x, h1, c1 = self.lstm1(x, initial_state=lstm_1_states, training=training)
            x = self.dropout1(x)
            x, h2, c2 = self.lstm2(x, initial_state=lstm_2_states, training=training)
            x = self.dense(x)
            if return_state:
                return x, c1, h1, c2, h2
            return x
        except:
            lstm_1_states = self.lstm1.get_initial_state(x)
            lstm_2_states = self.lstm2.get_initial_state(x)
            x, h1, c1 = self.lstm1(x, initial_state=lstm_1_states, training=training)
            x = self.dropout1(x)
            x, h2, c2 = self.lstm2(x, initial_state=lstm_2_states, training=training)
            x = self.dense(x)
            if return_state:
                return x, c1, h1, c2, h2
            return x
    
    @tf.function
    def getInitialState(self, batch_size):
        c1, h1 = self.lstm1.get_initial_state(tf.zeros((batch_size, 512)))
        c2, h2 = self.lstm2.get_initial_state(tf.zeros((batch_size, 512)))
        return c1, h1, c2, h2
