# garment-catalog-classifier
Clasificador de prendas en base al entrenamiento del modelo que proyecte la imagen de la prenda predicha y la prenda real 
# Introducción 
Fashion-MNIST es un conjunto de datos de imágenes de moda que contiene 70,000 imágenes en escala de grises de 28x28 píxeles, con 60,000 imágenes de entrenamiento y 10,000 imágenes de prueba. Cada imagen está etiquetada con una de las 10 clases posibles.
# Evolución de  Fashion-MNIST
Se creó Fashion-MNIST como un conjunto de datos más moderno y desafiante que contiene imágenes de moda en lugar de dígitos escritos a mano. Fashion-MNIST se considera un conjunto de datos más realista y moderno que MNIST y se espera que ayude a los investigadores y los practicantes del aprendizaje automático a desarrollar y evaluar algoritmos más avanzados para la clasificación de imágenes. Además, la creación de Fashion-MNIST también ha fomentado una mayor exploración de conjuntos de datos alternativos y más desafiantes para la clasificación de imágenes en el futuro.
# Técnicas avanzadas de aprendizaje automático para investigadores experimentados: perspectivas y desafíos
Hay varias buenas razones por las cuales los investigadores serios de aprendizaje automático están considerando reemplazar el conjunto de datos MNIST con Fashion-MNIST. Aquí hay algunas de las razones más importantes:
* Representatividad: El conjunto de datos MNIST se ha utilizado durante mucho tiempo como un estándar de facto para la evaluación de algoritmos de aprendizaje automático en la clasificación de imágenes. Sin embargo, debido a que MNIST contiene imágenes de dígitos escritos a mano, se ha criticado por ser poco representativo de las imágenes del mundo real. En contraste, Fashion-MNIST contiene imágenes de moda, lo que lo hace más representativo y desafiante para los algoritmos de clasificación de imágenes.
* Variedad: MNIST solo contiene imágenes de dígitos escritos a mano en blanco y negro, lo que lo hace muy limitado en términos de variedad de imágenes. Fashion-MNIST, por otro lado, contiene imágenes de moda de diferentes categorías y colores, lo que proporciona una mayor variedad y desafío para los algoritmos de clasificación de imágenes.
* Dificultad: Debido a que MNIST es un conjunto de datos relativamente fácil, los algoritmos de clasificación de imágenes han alcanzado un rendimiento casi perfecto en MNIST. Esto ha hecho que MNIST sea menos útil para evaluar y comparar el rendimiento de los algoritmos de clasificación de imágenes más avanzados. Fashion-MNIST, por otro lado, es un conjunto de datos más desafiante que puede proporcionar una mejor evaluación y comparación de los algoritmos de clasificación de imágenes más avanzados.
# Adquirir información en base a datos 
Hay varias formas de obtener los datos de Fashion-MNIST para su uso en proyectos de aprendizaje automático. Aquí hay algunas opciones:
Usar bibliotecas de ML: muchas bibliotecas populares de aprendizaje automático, como Tensorflow, Keras, PyTorch y scikit-learn, incluyen Fashion-MNIST como un conjunto de datos predeterminado o proporcionan API para descargarlo fácilmente. Esto facilita la obtención de los datos y su integración en proyectos de aprendizaje automático.

Una vez que descargue los archivos, puede cargar los datos utilizando las bibliotecas de Python, como numpy y struct, para leer los archivos y convertirlos en matrices de datos que se pueden utilizar en proyectos de aprendizaje automático.
 
# Dataset: fashion_mnist.load_data()

| Nombre | Descripción | Tamaño |
|:--------------|:-------------:|--------------:|
| train-images-idx3-ubyte.gz | Imágenes del conjunto de entrenamiento  | 26 MBytes |
| train-labels-idx1-ubyte.gz | Etiquetas de conjuntos de entrenamiento  | 29 KBytes |
| t10k-images-idx3-ubyte.gz | Imágenes del conjunto de pruebas  | 4,3 MBytes |
| t10k-labels-idx1-ubyte.gz | Etiquetas de conjuntos de prueba  | 5,1 KBytes |

# Clonar el repositorio de GitHub de Fashion-MNIST 
Otra opción para obtener los datos del conjunto de datos. Este repositorio incluye tanto el conjunto de datos de entrenamiento como el de prueba en la carpeta "data/fashion", y también proporciona algunos scripts útiles para la evaluación comparativa y visualización de los resultados de aprendizaje automático.Para clonar el repositorio, debe tener instalado Git en su sistema. 

git clone git@github.com:zalandoresearch/fashion-mnist.git

Esto descargará todo el repositorio en su directorio de trabajo actual. Luego, puede acceder a la carpeta de datos y utilizar los archivos de datos en su proyecto de aprendizaje automático.
Es importante tener en cuenta que clonar el repositorio es una buena opción si también está interesado en explorar otros aspectos del conjunto de datos, como los detalles de la implementación y la evaluación comparativa. Si solo está interesado en obtener los datos para usar en su proyecto de aprendizaje automático, es posible que desee considerar otras opciones, como el uso de bibliotecas de ML o la descarga directa.

# Prendas 
Cada ejemplo en el conjunto de datos Fashion-MNIST se asigna a una de las siguientes etiquetas:

| Etiqueta | Descripción | 
|:--------------|:-------------:|
| 0| Camiseta  | 
| 1 | Pantalón | 
| 2 | Jersey | 
| 3 | Vestido  | 
| 4 | Abrigo  | 
| 5 | Sandalia  | 
| 6 | Camisa  | 
| 7| Sneaker  | 
| 8 | Bolsa  | 
| 9 | Botín  |

Estas etiquetas se utilizan para identificar la clase de cada imagen de moda en el conjunto de datos. Es importante tener en cuenta que el objetivo del conjunto de datos es permitir a los investigadores evaluar algoritmos de aprendizaje automático para la clasificación de imágenes de moda en diez categorías diferentes. Por lo tanto, es importante utilizar estas etiquetas para evaluar la precisión de los modelos de aprendizaje automático en la tarea de clasificación de imágenes de moda.

# Formas comunes de cargar datos (Python)

Es importante tener en cuenta que este código requiere que la biblioteca NumPy esté instalada. Si aún no ha instalado NumPy, puede hacerlo fácilmente utilizando el administrador de paquetes pip.

```
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
```

En el código, x_train y y_train contienen las imágenes y las etiquetas del conjunto de datos de entrenamiento, respectivamente, y x_test e y_test contienen las imágenes y las etiquetas del conjunto de datos de prueba, respectivamente. La función load_mnist toma dos argumentos: la ruta al directorio que contiene los archivos de datos del conjunto de datos Fashion-MNIST (data/fashion en este caso), y el tipo de conjunto de datos que se va a cargar ('train' para el conjunto de datos de entrenamiento y 't10k' para el conjunto de datos de prueba).

# Convertimos los datos y normalizamos 

```
x_train = x_train.astype ('float32')
x_test=x_test.astype('float32')
x_train/=255
x_test/=255
```

La primera línea convierte el conjunto de entrenamiento x_train en números de punto flotante de 32 bits. 

La segunda línea hace lo mismo para el conjunto de prueba x_test.

La tercera línea normaliza los valores de los píxeles de x_train dividiéndolos por 255, que es el valor máximo posible de un píxel en una imagen. La cuarta línea hace lo mismo para el conjunto de prueba x_test.

La normalización de los valores de los píxeles en un rango de 0 a 1 puede ayudar a mejorar la eficacia del modelo, ya que asegura que todos los datos de entrada se encuentren en la misma escala.

# Definir el modelado de la prediccion 

```
model=Sequential()
model.add(Conv2D(32, kernel_size=(3,3),
                activation = 'relu',
                input_shape=input_shape))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
```

El código define un modelo de red neuronal convolucional (CNN) utilizando la API secuencial de Keras. El modelo consta de varias capas:

La primera capa es una capa convolucional con 32 filtros de tamaño 3x3 y función de activación ReLU. La entrada es una imagen con la forma especificada por input_shape.

La segunda capa es otra capa convolucional con 64 filtros de tamaño 3x3 y función de activación ReLU.

La tercera capa es una capa de agrupación máxima (max pooling) con un tamaño de ventana de 2x2.

La cuarta capa es una capa de abandono (dropout) con una tasa de abandono del 25%.

La quinta capa es una capa de aplanamiento (flatten) que convierte la salida de la capa anterior en un vector unidimensional.

La sexta capa es una capa densa (fully connected) con 128 neuronas y función de activación ReLU.

La séptima capa es otra capa de abandono con una tasa de abandono del 50%.

La octava y última capa es una capa densa con un número de neuronas igual al número de clases num_classes y función de activación softmax. Esta capa produce la salida final del modelo, que es una distribución de probabilidad sobre las posibles clases.

Este modelo es una CNN con dos capas convolucionales, una capa de agrupación máxima, dos capas de abandono, dos capas densas y una capa de salida con activación softmax.

# Entrenamos el modelo 

```
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.1)
```

El código entrena el modelo previamente compilado utilizando el método fit de Keras. Se especifican cuatro argumentos:

x_train: Este es el conjunto de datos de entrenamiento de entrada.

y_train: Este es el conjunto de datos de entrenamiento de salida esperada.

batch_size: Este es el número de muestras que se utilizarán en cada iteración del algoritmo de optimización. En este caso, se utiliza un tamaño de lote de 64.

epochs: Este es el número de épocas (iteraciones completas a través del conjunto de datos de entrenamiento) que se utilizarán para entrenar el modelo. En este caso, se utiliza un número de épocas de 10.

validation_split: Este argumento especifica la proporción de los datos de entrenamiento que se utilizarán como conjunto de validación. En este caso, se utiliza una proporción del 10%, lo que significa que el 10% de los datos de entrenamiento se utilizarán para la validación durante el entrenamiento.

Este código entrena el modelo utilizando el conjunto de datos de entrenamiento y validación, con un tamaño de lote de 64, 10 épocas de entrenamiento y una proporción del 10% de los datos de entrenamiento utilizados para la validación.

# Evaluamos el modelo en el conjunto de prueba

```
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)
```

El código evalúa el modelo previamente entrenado utilizando el conjunto de datos de prueba utilizando el método evaluate de Keras. Se almacenan los resultados de la evaluación en las variables test_loss y test_acc.

x_test: Este es el conjunto de datos de prueba de entrada.

y_test: Este es el conjunto de datos de prueba de salida esperada.

El método evaluate calcula la función de pérdida y las métricas especificadas durante la compilación del modelo en el conjunto de datos de prueba y devuelve los resultados de la evaluación.

Finalmente, se imprime el valor de la precisión (accuracy) del modelo en el conjunto de datos de prueba utilizando la variable test_acc.
Este código evalúa el modelo en el conjunto de datos de prueba y muestra la precisión obtenida en el mismo.

# Predicciones 

```
predictions = model.predict(x_test)
```

El código utiliza el modelo previamente entrenado para hacer predicciones sobre el conjunto de datos de prueba utilizando el método predict de Keras. Se almacena el resultado de las predicciones en la variable predictions.

x_test: Este es el conjunto de datos de prueba de entrada.

El método predict toma el conjunto de datos de prueba de entrada y devuelve las predicciones de salida correspondientes generadas por el modelo entrenado. En este caso, se utilizó el modelo de clasificación de imágenes para predecir las clases de las imágenes en el conjunto de datos de prueba.

Este código utiliza el modelo previamente entrenado para hacer predicciones sobre el conjunto de datos de prueba y almacena los resultados en la variable predictions.

# Clasificacion de prendas y visualización de prendas erroneas  

```
    import matplotlib.pyplot as plt
n=0
for i in range(25):
    predicted_label = np.argmax(predictions[i])
    true_label = np.argmax( y_test[i])
    if true_label != predicted_label:
        plt.figure(figsize=(10,10))
        n=n+1
        plt.subplot(5,5,n)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.title(f"prenpred: {tipoprenda.get(predicted_label)} prenreal: {tipoprenda.get(true_label)} ")
        plt.imshow(x_test[i], cmap=plt.cm.binary)
        plt.tight_layout()
        plt.show()
```

El código muestra visualmente algunas de las predicciones erróneas que hizo el modelo. Se utiliza la biblioteca Matplotlib para crear una cuadrícula de imágenes, donde cada imagen es una predicción errónea.

n: Esta variable se utiliza para contar el número de predicciones erróneas que se muestran.

predicted_label: Esta variable almacena la etiqueta predicha para la imagen actual en el bucle.

true_label: Esta variable almacena la etiqueta verdadera para la imagen actual en el bucle.

plt.figure(): Este método crea una nueva figura de tamaño 10x10 para mostrar las imágenes de las predicciones erróneas.

plt.subplot(): Este método crea una subtrama dentro de la figura, donde se mostrará una imagen. La función tight_layout() se utiliza para ajustar la disposición de las subtramas en la figura.

plt.grid(): Este método desactiva las líneas de la cuadrícula en la imagen.

plt.xticks() y plt.yticks(): Estos métodos desactivan las etiquetas de los ejes x e y en la imagen.

plt.title(): Este método agrega un título a la imagen que muestra la prenda real y la prenda predicha por el modelo.

plt.imshow(): Este método muestra la imagen actual en la subtrama actual.

Este código muestra visualmente algunas de las predicciones erróneas que hizo el modelo, junto con la prenda real y la prenda predicha por el modelo. Se utiliza la biblioteca Matplotlib para crear una cuadrícula de imágenes para mostrar las predicciones erróneas.

<img width="164" alt="Captura 1" src="https://user-images.githubusercontent.com/126996071/222946476-3d0cae42-253c-459c-9b54-e650ff5be091.PNG">
<img width="178" alt="Captura 2" src="https://user-images.githubusercontent.com/126996071/222946477-50028c76-8c23-48ed-89ab-a88d30133d5c.PNG">
<img width="183" alt="Captura 3" src="https://user-images.githubusercontent.com/126996071/222946479-41c264cd-6615-43ae-88ad-4b73ed274213.PNG">
<img width="181" alt="Captura 4" src="https://user-images.githubusercontent.com/126996071/222946480-f3f14a44-119a-43ce-a643-20a6913ea2f7.PNG">
