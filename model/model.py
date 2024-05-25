from tensorflow.keras import models, layers, applications

def create_model():
    # Criar o modelo base VGG16 pré-treinado
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Congelar as camadas do modelo base
    for layer in base_model.layers:
        layer.trainable = False

    # Adicionar camadas densas personalizadas
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    return model


