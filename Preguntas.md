
# Preguntas sobre el ejemplo de clasificación de imágenes con PyTorch y MLP

## 1. Dataset y Preprocesamiento
- ¿Por qué es necesario redimensionar las imágenes a un tamaño fijo para una MLP?

Las entradas asociadas a una MLP son de una dimension vectorial determinada. 
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

- ¿Qué ventajas ofrece Albumentations frente a otras librerías de transformación como `torchvision.transforms`?

Abluminations provee una variedad extremadamente amplia de tipos de transformaciones y extensiones de datasets que no provee tochvision.transforms, ademas de alteraciones y diferentes tipos de warping.

- ¿Qué hace `A.Normalize()`? ¿Por qué es importante antes de entrenar una red?

La normalizacion de un dataset mejora la generalizacion de una red neuronal y la velocidad de convergencia sobre un minimo en entrenamiento. La normalizacion de varianza y valor medio de las imagenes evita la saturacion de funciones de activacion y evade la posibilidad de gradientes sumamente bajas en valores fuera de ceirta cantidad de desviaciones estandar del promedio, especialmente en la etapa inicial de entrenamiento.

- ¿Por qué convertimos las imágenes a `ToTensorV2()` al final de la pipeline?
Transforma de formato numpy (H, W, C) a formato aceptable por pytorch (C, H, W) de tipo float32. Para poder posteriormente ser efectivamente transformado en un formato de entrada permitido por la MLP.

## 2. Arquitectura del Modelo
- ¿Por qué usamos una red MLP en lugar de una CNN aquí? ¿Qué limitaciones tiene?

MLP en principio no posee invarianza sobre shifteos, rotaciones, y en general cualquier tipo de modificaciones sobre la entrada. CONTINUAR

- ¿Qué hace la capa `Flatten()` al principio de la red?
Transformacion del tensor de tamano (C, H, W), en un vector de tamano (C x H x W).

- ¿Qué función de activación se usó? ¿Por qué no usamos `Sigmoid` o `Tanh`?

El modelo simple implementado hace uso de una funcion RELU(x) = max[0, x]. 
Tiene muchisima ventaja frente a las funciones sigmoide y tangente hiperbolico.
1. Su naturaleza matematica implica una eficiencia computacional muchisimo mayor a la sigmoidal y tangente hiperbolica (Funciones que hacen uso de la funcion exponencial, contra una funcion de comparacion de menor costo computacional).
2. La funcion RELU presenta un efecto de "Sparsity": Esto es, logra que para una amplia cantidad de neuronas su salida sea cero. Funciona estrictamente como un "Filtro", donde para cierto tipo de pixeles, la funcion RELU permite instantaneamente lograr una eliminacion de efectuar un cambio sobre la salida dado dicha entrada.
3. Provee una amplia ventaja en el entrenamiento, ya que su derivada puede ser "0" o "1". Las funciones Tanh y sigmoide poseen derivadas no acotadas, lo que puede resultar en una "Gradient Vanishing" (Desaparicion de Gradiente) sobre los parametros, lo que implica un entrenamiento suboptimo en el caso de que toda derivada posea un valor bajo. Este caso es muchisimo menos probable al utilizar neuronas con funcion de activacion de RELU, ya que cada una de ellas deberia estar "apagada"; Al utilizar una MLP con una cantidad de neuronas elevadas, la desaparicion de gradiente ya es un problema basicamente evadido.


- ¿Qué parámetro del modelo deberíamos cambiar si aumentamos el tamaño de entrada de la imagen?

xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

## 3. Entrenamiento y Optimización
- ¿Qué hace `optimizer.zero_grad()`?

Resetea los gradientes almacenados en el optimizador. Si no lo hiciéramos, se acumularían de iteraciones previas.

- ¿Por qué usamos `CrossEntropyLoss()` en este caso?

El problema a resolver con este MLP se resume en la clasificacion Multiclase de un dataset. La funcion de perdida de Cross Entropy es clave para encarar dicho problema. Su impelemntacion combina las ventajas de LogSoftmax y NLL en una funcion.

- ¿Cómo afecta la elección del tamaño de batch (`batch_size`) al entrenamiento?

- ¿Qué pasaría si no usamos `model.eval()` durante la validación?
El loop de entrenamiento no funcionaria como tal. El modelo no se evaluaria como tal con los nuevos parametros calculados en el ulitmo batch, y el MLP estaria basicamente estancado en un conjunto de parametros invariantes, generados en la primer iteracion de entrenamiento. 

## 4. Validación y Evaluación
- ¿Qué significa una accuracy del 70% en validación pero 90% en entrenamiento?

Muestra un caso tipico de OVERFITTING. Aunque no es necesariamente el caso, por lo general esto demuestra que el modelo esta estrictamente entrenado/acotado a un subespacio muestral de todo el universo de casos posibles, que no generaliza dicho universo de predicciones.

- ¿Qué otras métricas podrían ser más relevantes que accuracy en un problema real?

La diferencia en importancia en metricas recide directamente en el peligro de cometer un falso positivo contra un falso negativo. Caracterizar al modelo con una metrica que perciba el peso e importancia de cada error es sumamente importante en las aplicaciones del mundo real (medicina, militar, etc.).

La precision de un modelo provee una metrica de la accuracy de las predicciones positivas correctas sobre una clase, con respecto a la cantidad total de predicciones positivas (Correctas e incorrectas) realizadas sobre esa clase (True positive / (True positive + False Positive)).

La metrica Recall en cambio, prove una metrica del accuracy de las predicciones correctas sobre una clase, con respecto a la cantidad de predicciones correctas sobre esa clase + la cantidad de predicciones hechas de manera incorrecta sobre esa clase (True Positive / (True positive + False Negative))

f1-score es un promedio armonico entre la precision y el recall. Cuando tanto el error del falso positivo como falso negativo tienen un peso similar, el f1-score es una metrica de alta calidad.

- ¿Qué información útil nos da una matriz de confusión que no nos da la accuracy?

La matriz de confusion es ideal para problemas de clasificacion multiclase. La accuracy es una metrica adimensional, que no provee informacion detallada sobre donde esta fallando el modelo; es decir, como se distribuye el error en accuracy en las predicciones realizadas. La matriz de confusion resuelve justamente dicho problema. En las filas, muestra el "true value", y en las columnas el "predicted value", de forma tal que es posible encontrar cuales son las clases mas problematicas en el modelo. Muchisima informacion estadistica sobre los errores cometidos en cada clase puede ser extraida de dicha matriz.

- En el reporte de clasificación, ¿qué representan `precision`, `recall` y `f1-score`?

(Respuesta de pregunta 1)

## 5. TensorBoard y Logging 
- ¿Qué ventajas tiene usar TensorBoard durante el entrenamiento?



- ¿Qué diferencias hay entre loguear `add_scalar`, `add_image` y `add_text`?



- ¿Por qué es útil guardar visualmente las imágenes de validación en TensorBoard?



- ¿Cómo se puede comparar el desempeño de distintos experimentos en TensorBoard?




## 6. Generalización y Transferencia
- ¿Qué cambios habría que hacer si quisiéramos aplicar este mismo modelo a un dataset con 100 clases?

El codigo resulta estructurado con la generalidad suficiente de forma tal que al extender el dataset con el mismo formato utilizado, a uno con 100 clases, y la leve modificacion en la clase:
class MLPClassifier(nn.Module):
    def __init__(self, input_size=64*64*3, num_classes=***100***):

Deberia ser suficiente para adaptar el modelo a una cantidad mas amplia de clases.

- ¿Por qué una CNN suele ser más adecuada que una MLP para clasificación de imágenes?

Las redes neuronales convolucionales (CNNs) son más adecuadas que las MLPs para clasificación de imágenes porque aprovechan la estructura espacial de los datos. A diferencia de las MLPs, que tratan cada píxel como una entrada independiente, las CNNs trabajan con regiones locales de la imagen y conservan relaciones espaciales. Utilizan filtros convolucionales que detectan patrones como bordes, texturas y formas básicas en distintas posiciones. Estos filtros comparten pesos, lo que reduce significativamente la cantidad de parámetros y mejora la eficiencia del modelo. Además, esta estructura permite que las CNNs sean naturalmente invariantes a traslaciones, es decir, que puedan reconocer un objeto aunque cambie de ubicación. A medida que se avanza en las capas, las CNNs construyen representaciones jerárquicas que capturan características más abstractas. Esto favorece una mejor generalización y una mayor capacidad para aprender conceptos complejos. Por estas razones, las CNNs son hoy la arquitectura de referencia en tareas de visión por computadora y clasificacion de imagenes.

- ¿Qué problema podríamos tener si entrenamos este modelo con muy pocas imágenes por clase?

Overfitting es un problema extremadamente comun cuando se hace uso de un dataset lo suficientemente pequeno. 

- ¿Cómo podríamos adaptar este pipeline para imágenes en escala de grises?

xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


## 7. Regularización

### Preguntas teóricas:
- ¿Qué es la regularización en el contexto del entrenamiento de redes neuronales?

La regularización es un conjunto de técnicas utilizadas durante el entrenamiento para evitar que el modelo se adapte demasiado a los datos de entrenamiento, topologia de la red, y caracteristicas de hiperparametros, logrando así que generalice mejor a datos nuevos o no vistos.

- ¿Cuál es la diferencia entre `Dropout` y regularización `L2` (weight decay)?

Dropout es una tecnica de regularizacion que penaliza la sobre-dependencia de una topologia neurologica sobre sub-grafos / seccionees de la red o neuronas especificas para realizar predicciones. La tecnica consta en desactivar cierto conjunto de neuronas dentro de la red en cada iteracion de entrenamiento. Existen multiples algoritmos para la seleccion de las "Dropout neurons", pero una forma clasica recide en desactivar las neuronas con una probabilidad p en cada iteracion de entrenamiento.

La tecnica de regularizacion L2 es una tecnica que aplica una penalización al valor de los pesos grandes. Esto fuerza al modelo a desarrollarse con pesos pequenos, lo que implica funciones de seleccion mas suaves, y reduce la incidencia al overfitting.

- ¿Qué es `BatchNorm` y cómo ayuda a estabilizar el entrenamiento?
- ¿Cómo se relaciona `BatchNorm` con la velocidad de convergencia?
- ¿Puede `BatchNorm` actuar como regularizador? ¿Por qué?

Batch norm es una tecnica de regularizacion que normaliza la entrada a las funciones de activacion en cada neurona. Batch Normalization normaliza las activaciones intermedias de cada capa para que tengan media cero y varianza uno, lo que estabiliza y acelera el entrenamiento. Esto evita que las distribuciones de activación cambien drásticamente entre batches, permitiendo usar learning rates más altos sin perder estabilidad. Además, al introducir pequeñas variaciones en las estadísticas de cada batch, actúa como una forma de regularización implícita que reduce el sobreajuste. También mantiene las activaciones en rangos donde las funciones no lineales, como ReLU o tanh, no se saturan, favoreciendo un flujo de gradientes más saludable. En resumen, BatchNorm mejora la velocidad, estabilidad y capacidad de generalización de las redes neuronales.
Sí, BatchNorm puede actuar como regularizador porque introduce variación entre batches durante el entrenamiento, forzando a la red a ser más robusta.En la práctica, reduce el sobreajuste, y en algunos casos, incluso permite prescindir de Dropout.

- ¿Qué efectos visuales podrías observar en TensorBoard si hay overfitting?

xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

- ¿Cómo ayuda la regularización a mejorar la generalización del modelo?

xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

### Actividades de modificación:
1. Agregar Dropout en la arquitectura MLP:
   - Insertar capas `nn.Dropout(p=0.5)` entre las capas lineales y activaciones.

   - Comparar los resultados con y sin `Dropout`.

2. Agregar Batch Normalization:
   - Insertar `nn.BatchNorm1d(...)` después de cada capa `Linear` y antes de la activación:
     ```python
     self.net = nn.Sequential(
         nn.Flatten(),
         nn.Linear(in_features, 512),
         nn.BatchNorm1d(512),
         nn.ReLU(),
         nn.Dropout(0.5),
         nn.Linear(512, 256),
         nn.BatchNorm1d(256),
         nn.ReLU(),
         nn.Dropout(0.5),
         nn.Linear(256, num_classes)
     )
     ```

3. Aplicar Weight Decay (L2):
   - Modificar el optimizador:
     ```python
     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
     ```

4. Reducir overfitting con data augmentation:
   - Agregar transformaciones en Albumentations como `HorizontalFlip`, `BrightnessContrast`, `ShiftScaleRotate`.

5. Early Stopping (opcional):
   - Implementar un criterio para detener el entrenamiento si la validación no mejora después de N épocas.

### Preguntas prácticas:
- ¿Qué efecto tuvo `BatchNorm` en la estabilidad y velocidad del entrenamiento?
- ¿Cambió la performance de validación al combinar `BatchNorm` con `Dropout`?
- ¿Qué combinación de regularizadores dio mejores resultados en tus pruebas?
- ¿Notaste cambios en la loss de entrenamiento al usar `BatchNorm`?

## 8. Inicialización de Parámetros

### Preguntas teóricas:
- ¿Por qué es importante la inicialización de los pesos en una red neuronal?
- ¿Qué podría ocurrir si todos los pesos se inicializan con el mismo valor?
- ¿Cuál es la diferencia entre las inicializaciones de Xavier (Glorot) y He?
- ¿Por qué en una red con ReLU suele usarse la inicialización de He?
- ¿Qué capas de una red requieren inicialización explícita y cuáles no?

### Actividades de modificación:
1. Agregar inicialización manual en el modelo:
   - En la clase `MLP`, agregar un método `init_weights` que inicialice cada capa:
     ```python
     def init_weights(self):
         for m in self.modules():
             if isinstance(m, nn.Linear):
                 nn.init.kaiming_normal_(m.weight)
                 nn.init.zeros_(m.bias)
     ```

2. Probar distintas estrategias de inicialización:
   - Xavier (`nn.init.xavier_uniform_`)
   - He (`nn.init.kaiming_normal_`)
   - Aleatoria uniforme (`nn.init.uniform_`)
   - Comparar la estabilidad y velocidad del entrenamiento.

3. Visualizar pesos en TensorBoard:
   - Agregar esta línea en la primera época para observar los histogramas:
     ```python
     for name, param in model.named_parameters():
         writer.add_histogram(name, param, epoch)
     ```

### Preguntas prácticas:
- ¿Qué diferencias notaste en la convergencia del modelo según la inicialización?
- ¿Alguna inicialización provocó inestabilidad (pérdida muy alta o NaNs)?
- ¿Qué impacto tiene la inicialización sobre las métricas de validación?
- ¿Por qué `bias` se suele inicializar en cero?