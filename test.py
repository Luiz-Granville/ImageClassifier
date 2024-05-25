import requests
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing import image
import numpy as np

# Carregar o conjunto de dados CIFAR-10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Selecionar uma imagem de teste e seu rótulo correspondente
img_array = test_images[0]
img_label = test_labels[0]

# Salvar a imagem como um arquivo temporário 'test_image.jpg'
img = image.array_to_img(img_array)
img.save('images/test_image.jpg')

# Testar a imagem usando a API
url = 'http://127.0.0.1:5000/predict'
files = {'image': open('test_image.jpg', 'rb')}
response = requests.post(url, files=files)

# Exibir a classe prevista e a classe real
print(f"Predicted class: {response.json()['predicted_class']}, Actual class: {img_label[0]}")
