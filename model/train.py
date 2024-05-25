from tensorflow.keras import datasets, models, layers, callbacks
import utils
import model

# Carregar CIFAR-10
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalizar os dados
train_images, test_images = utils.normalize_data(train_images, test_images)

# Criar o modelo
model = model.create_model()

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



# Definir callbacks
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Treinar o modelo
model.fit(train_images, train_labels, epochs=50, validation_data=(test_images, test_labels), callbacks=[early_stopping])

# Salvar o modelo
model.save('model/model_weights.h5')
