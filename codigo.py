import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layres import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.processing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# definir o modelo
def criar_modelo():
    modelo = Sequential([    
        #primeira camada convolucional
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),

        #segunda camada convolucional
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        #terceira camada convolucional
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2, 2),

        #achatar os dados para as camadas densas
        Flatten(),

        #camada densa
        Dense(512, activation='relu'),
        Dropout(0.5), #reduzir overfitting
        Dense(2, activation='softmax') # 4 classes: gato, cachorro
    ])

    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo

# Preparar os dados
def preparar_dados(diretorio_treino, diretorio_validacao):
    #geradores de dados para treino e validação
    gerador_treino = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    gerador_validacao = ImageDataGenerator(rescale='1./255')

    #carregando as imagens das pastas
    conjutno_treino = gerador_treino.flow_from_directory(
        diretorio_treino,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    conjunto_validacao = gerador_validacao.flow_from_directory(
        diretorio_validacao,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    return conjutno_treino, conjunto_validacao

# treinar modelo
def treinar_modelo(modelo, conjunto_treino, conjunto_validacao, epocas=10):
    historico = modelo.fit(
        conjunto_treino,
        steps_per_epoch=conjunto_treino.samples // conjunto_treino.batch_size,
        epochs=epocas,
        validation_data=conjunto_validacao,
        validation_steps=conjunto_validacao.samples // conjunto_validacao.batch_size
    )

    return historico

# Salvar modelo treinado
def salvar_modelo(modelo, caminho='modelo_classificador_animais.h5'):
    modelo.save(caminho)
    print(f'Modelo salvo em {caminho}')

# Funcao para classificar uma nova imagem
def classificar_imagem(caminho_imagem, modelo):
    img = image.load_img(caminho_imagem, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0 # normalizar a imagem

    # fazer a previsão
    predicao = modelo.predict(img)

    # mapear indices para nome de classes
    classes = ['gato', 'cachorro']

    # obter a classe com maior probabilidade
    classe_predita = classes[np.argmax(predicao)]
    probabilidade = np.max(predicao) * 100

    # mostrar imagem e a previsao
    plt.imshow(img)
    plt.title(f'Previsão: {classe_predita} ({probabilidade:.2f}%)')
    plt.axis('off')
    plt.show()

    #mostrar todas as probalidades
    for i, classe in enumerate(classes):
        print(f'{classe}: {predicao[0][i] * 100:.2f}%')

    return classe_predita, probabilidade

# Exemplo de uso
if __name__ == "__main__":