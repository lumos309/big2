import tensorflow as tf
import joblib
import handPredictionModel

x_train = joblib.load('handPredictionInputs')
y_train = joblib.load('handPredictionTargets')

MAX_GAME_LENGTH = 25
for i in range(len(x_train)):
    if len(x_train[i]) > MAX_GAME_LENGTH:
        x_train[i] = x_train[i][:MAX_GAME_LENGTH]
        y_train[i] = x_train[i][:MAX_GAME_LENGTH]
    elif len(x_train[i]) < MAX_GAME_LENGTH:
        for j in range(MAX_GAME_LENGTH - len(x_train[i])):
            x_train[i].append([0 for m in range(211)])
            y_train[i].append([0 for m in range(156)])

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True)
model = handPredictionModel.HandPredictionModel()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=tf.losses.BinaryCrossentropy())

x_test = x_train[0]
x_train = tf.convert_to_tensor(x_train, dtype=float)
y_train = tf.convert_to_tensor(y_train, dtype=float)
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = (
    dataset
        .shuffle(1000, reshuffle_each_iteration=True)
        .batch(64, drop_remainder=True)
)

print(dataset)

EPOCHS = 1
history = model.fit(dataset, epochs=EPOCHS)
tf.saved_model.save(model, 'handPredictionModel')
model.save_weights('handPredictionModelWeights')