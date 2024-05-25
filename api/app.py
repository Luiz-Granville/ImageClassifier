from flask import Flask, request, Response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import json

app = Flask(__name__)
model = load_model('model/model_weights.h5')

class_names = ["Airplane (Avião)", "Automobile (Automóvel)", "Bird (Pássaro)",
               "Cat (Gato)", "Deer (Veado)", "Dog (Cachorro)", "Frog (Sapo)",
               "Horse (Cavalo)", "Ship (Navio)", "Truck (Caminhão)"]

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return Response(response=json.dumps({'error': 'No image file found in the request'}, ensure_ascii=False), status=400, content_type='application/json; charset=utf-8')

    img_file = request.files['image']
    img = image.load_img(io.BytesIO(img_file.read()), target_size=(32, 32))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]

    return Response(response=json.dumps({'predicted_class': predicted_class_name}, ensure_ascii=False), status=200, content_type='application/json; charset=utf-8')

if __name__ == '__main__':
    app.run(debug=True)
