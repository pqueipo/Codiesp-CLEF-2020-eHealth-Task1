# Dictionary based fuzzy matching with SpaCy named entity recognition

El diccionario de ICD-10-CM contiene 98288 códigos de diagnóstico válidos. Contiene los siguientes campos:
code: códigos
es-description: descripción en español
en-description: descripción en inglés

Se puede reconocer entidades en español e inglés utilizando la relación entre el código, la descripción en español o inglés y el contenido de las historias clínicas.
Se produce un emparejamiento de sintagmas nominales y términos equivalentes del diccionario ICD-10. Estos recursos se utilizan en gran cantidad de trabajos para tareas
de reconocimiento de entidades biomédicas. Los enfoques de correspondencia exacta alcanzan alta precisión generalmente, aunque sufren de menores recuperaciones debido a
variaciones léxicas, abreviaturas u errores ortográficos. 

Para aumentar la recuperación, los enfoques de coincidencia por aproximación utilizan similitud léxica y coincidencia de cadenas de caracteres por aproximación.

## Importar corpus
Se han descargado y preparado los directorios del corpus. Se han combinado todos los ficheros de texto presentes en una carpeta como un fichero único en el script «files_to_tsv».

## Preprocesado de datos
Es importante preparar los datos de entrada antes y después de aplicar procesos en las historias de caso clínico para obtener el output esperado. Algunas tareas de preprocesamiento son la limpieza de duplicados, normalización, combinación de ficheros y tokenización en palabras y signos de puntuación.

## Pipeline de procesamiento de lenguaje de SpaCy
Se ha utilizado la librería Spacy para realizar tareas de procesamiento de la nota y reconocimiento de entidades de diagnóstico clínico. Tanto el modelo en español es-core-news-sm (2.2.5) y el modelo en inglés encorewebsm (2.2.5) incluyen capas convolucionales, conexiones residuales, normalización de capas y maxout non-linearity.

Se ha creado una línea de procesos reutilizando elementos de SpaCy que incluye tagger, parser y entity recognizer (reconocedor de entidades nombradas). Se ha iterado sobre
las fases del pipeline. Se crean componentes con «create_pipe» y se añade un componente a la línea de procesos con «add_pipe». También se ha creado un función personalizada.

## Comparación difusa de cadenas
Los métodos basados en diccionario son populares a la hora de abordar letras ambiguas. Los métodos de búsqueda en diccionario suelen ser robustos. Sin embargo, pueden
incluir cálculos complicados apriori ya que un tamaño grande de diccionario aumenta el coste de búsqueda. Por ello, la distancia Levenshtein es una métrica sencilla que
puede ser una herramienta de aproximación de cadenas efectiva.  La librería Fuzzy implementa algoritmos fonéticos en Python. Más específicamente, la librería Fuzzy Wuzzy realiza un mapeo de cadenas difuso. Para ello, utiliza la distancia de Levenshtein para calcular las diferencias entre secuencias. Gracias a esta librería, se pueden comparar las entidades detectadas en la historia clínica y buscar los códigos cuya descripción se aproxima más fielmente al contenido de la nota. La función de fuzzy wuzzy asigna una puntuación de similitud entre cadenas, y es posible configurar el umbral de permisividad, con una puntuación máxima y mínima. En este caso se ha elegido un máximo de -1 y un mínimo de 50.

 ## Predicción del conjunto test
Gracias a los pasos anteriores es posible realizar la predicción en español e inglés sobre el conjunto test (background + goldstandard) con 3001 historias de caso clínicas. Sin
embargo, solo se consideran las predicciones del conjunto goldstandard con 250 historias anotadas para la evaluación.




