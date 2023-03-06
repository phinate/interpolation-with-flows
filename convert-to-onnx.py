import keras
import keras2onnx

input_file = ""
output_file = ""

model = keras.models.load_model(input_file)
onnx_model = keras2onnx.convert_keras(model, model.name)
keras2onnx.save_model(onnx_model, output_file)

