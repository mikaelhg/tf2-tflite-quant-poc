import os
import pathlib
import tensorflow as tf
import logging

## --------------------------------------------------------------------------------
# Configure

try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  print('Cannot set memory growth when virtual devices configured')

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
  tf.config.experimental.set_virtual_device_configuration(
    physical_devices[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])

# tf.config.optimizer.set_jit(True)

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

## --------------------------------------------------------------------------------
# Train

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

checkpoint_path = "model/mnist_training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1)

model.fit(x_train, y_train, epochs=5, callbacks=[cp_callback])

model.evaluate(x_test,  y_test, verbose=2)

model.save('model/mnist_saved')

## --------------------------------------------------------------------------------
# Convert to TFLite

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_models_dir = pathlib.Path("model/mnist_tflite")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

tflite_model_file = tflite_models_dir / "mnist_model.tflite"
tflite_model_file.write_bytes(tflite_model)

## --------------------------------------------------------------------------------
# INT8

converter.optimizations = [tf.lite.Optimize.DEFAULT]

mnist_train, _ = tf.keras.datasets.mnist.load_data()
images = tf.cast(mnist_train[0], tf.float32)/255.0
mnist_ds = tf.data.Dataset.from_tensor_slices((images)).batch(1)
def representative_data_gen():
  for input_value in mnist_ds.take(100):
    yield [input_value]

converter.representative_dataset = representative_data_gen

tflite_model_quant = converter.convert()
tflite_model_quant_file = tflite_models_dir / "mnist_model_quant.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant)
