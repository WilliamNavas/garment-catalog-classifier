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

# convertimos los datos y normalizamos 

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

