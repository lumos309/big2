import tensorflow as tf
import time
import joblib
import handPredictionModel

#tf.compat.v1.disable_eager_execution()

class OneStep(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    @tf.function
    def predict(self, inputs, states_l1=None, states_l2=None):
        inputs = tf.expand_dims(tf.expand_dims(inputs, axis=0), axis=0)
        predicted_logits, c1, h1, c2, h2= self.model(x=inputs, lstm_1_states=states_l1, lstm_2_states=states_l2, return_state=True, training=False)
        predicted_logits = predicted_logits[:, -1, :]

        return predicted_logits, [c1, h1], [c2, h2]
    
    @tf.function
    def predictBatch(self, inputs, states_l1=None, states_l2=None):
        inputs = tf.cast(tf.expand_dims(inputs, axis=1), tf.float32)
        predicted_logits, c1, h1, c2, h2 = self.model(x=inputs, lstm_1_states=states_l1, lstm_2_states=states_l2, return_state=True, training=False)
        predicted_logits = predicted_logits[:, -1, :]

        return predicted_logits, [c1, h1], [c2, h2]

if __name__ == '__main__':
    handPredictionModel = handPredictionModel.HandPredictionModel()
    handPredictionModel.load_weights('handPredictionModelWeights')
    # model = OneStep(model)

    # model = tf.saved_model.load('oneStepModel')
    oneStepModel = OneStep(handPredictionModel)

    # tf.saved_model.save(oneStepModel, 'oneStepModel')
    # oneStepModel = tf.saved_model.load('oneStepModel')

    # states = None
    # for i in range(10):
    #     t = tf.zeros([211])
    #     pred, states = oneStepModel.predict(t, states)
    #     print(pred)
    x_train = joblib.load('handPredictionInputs')
    x_train = x_train[0]
    states = None
    for t in x_train:
        t = tf.convert_to_tensor(t, dtype=float)
        inputs = tf.stack([t, t, t])
        
        pred, states = oneStepModel.predictBatch(inputs, states)
        print(pred)