import numpy as np
import tensorflow as tf
from PIL import Image

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="beerDetectionModel.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read the sample JPG image and pre-process it for TensorFlow Lite
image = Image.open("test_img_01.jpg")
image = image.resize((150, 150), Image.ANTIALIAS)
image_data = np.array(image, dtype=np.float32)

# Set the input tensor to the image data
interpreter.set_tensor(input_details[0]['index'], [image_data])

# Run the model
interpreter.invoke()

# Get the output from the model
class_names = ["becks","berliner_kindl","corona","krombacher","krombacher_alkfrei","mönchshof","sterni","warsteiner"]
output_data = interpreter.get_tensor(output_details[0]['index'])
score = tf.nn.softmax(output_data)
print(
     "This image most likely belongs to {} with a {:.2f} percent confidence."
     .format(class_names[np.argmax(score)], 100 * np.max(score))
)

