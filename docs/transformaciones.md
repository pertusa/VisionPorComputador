
# Tema 3- Procesamiento de imagen: Transformaciones

En este tema comenzaremos a modificar imágenes mediante transformaciones de varios tipos.

## Transformaciones puntuales

En OpenCV se pueden realizar operaciones directas con matrices. Por ejemplo, dada una imagen `src` podemos multiplicar el valor de todos sus píxeles por 4 de esta forma:

```cpp
Mat dst = 4 * src;
```

Como puedes ver, en las operaciones aritméticas podemos usar tanto números como otras matrices. Si operamos con varias matrices debemos tener en cuenta que todas ellas deben tener la misma resolución radiométrica (_depth_) y deben ser del mismo tipo.

Además de las operaciones aritméticas básicas (suma, resta, multiplicación y división), también podemos usar _AND_, _OR_, _XOR_ y _NOT_ mediante las siguientes funciones:

```cpp
Mat res;
bitwise_and(src1, src2, dst);
bitwise_or(src2, src2, dst);  
bitwise_xor(src1, src2, dst);
bitwise_not(src1, dst);
```

Por ejemplo, para invertir una imagen (transformar lo blanco a negro y lo negro a blanco) podemos usar la instrucción `bitwise_not`.

Ecualizar histogramas en escala de grises es muy sencillo con la función `equalizeHist`, que necesita recibir la imagen original y la imagen donde guardaremos el resultado:

```cpp
Mat dst;
equalizeHist(src, dst);
```

También podemos umbralizar una imagen en escala de grises mediante la función [`threshold`](http://docs.opencv.org/2.4/doc/tutorials/imgproc/threshold/threshold.html), obteniendo como resultado una imagen binaria (también llamada máscara) que puede resaltar información relevante para una tarea determinada. La umbralización consiste en poner a 0 los píxeles que tienen un valor inferior al umbral indicado, y es la forma más básica de realizar segmentación (como veremos en detalle en el tema 5). Ejemplo de llamada a `threshold`:

```cpp
threshold(src, dst, 128, 255, CV_THRESH_BINARY); // Ponemos a 0 los píxeles cuyos valores estén por debajo de 128, y a 255 los que estén por encima.
```

El último parámetro es el tipo de umbralización. En OpenCV tenemos 5 tipos de umbralización que pueden consultarse [aquí](http://docs.opencv.org/2.4/doc/tutorials/imgproc/threshold/threshold.html), aunque el valor más usado es `CV_THRESH_BINARY` (umbralización binaria).

Para umbralizar imágenes en color, OpenCV ofrece la función `inrange`. Dada una imagen en 3 canales, esta función devuelve otra imagen de un canal que contiene en blanco aquellos píxeles que están en un determinado rango, y en negro los que quedan fuera del mismo. Por tanto, puede usarse para realizar una segmentación básica por color, tal como veremos en detalle en el tema 5.

```cpp
inRange(src, Scalar(0, 10, 20), Scalar(40, 40, 51), dst);
```

En OpenCV existen técnicas alternativas de binarización como el umbralizado adaptativo o el método de Otsu, que también veremos en el tema de segmentación porque no se pueden considerar transformaciones puntuales al tener en cuenta los valores de intensidad de los píxeles vecinos.

---

### Ejercicio

Haz un programa llamado `bct.cpp` que reciba 3 parámetros: El nombre de la imagen (que leeremos en escala de grises), un valor para el contraste (lo llamaremos _alpha_, de tipo `double`) y un valor para el brillo (_beta_, de tipo `int`). Debes comprobar que el programa recibe los parámetros indicados. Ejemplo de uso:

```bash
./bct lena.jpg 0.5 2
```

Con este ejemplo, el resultado debería ser como la siguiente imagen:

![Transformaciones brillo contraste](images/transformaciones/bct.jpg)



<!--- TODO: PONER IMAGEN SALIDA EJEMPLO --->


<!---
Cambiados rangos [0,3] por [0,2] y [0,100] por [-50,50]
--->

El programa debe guardar en un fichero llamado `bct.jpg` (siempre con este mismo nombre) el resultado de ajustar la imagen de entrada con el brillo y contraste indicados. Para hacer pruebas puedes usar valores de _alpha_ en el rango [0,2] y de _beta_ en el rango [-50,50].

<!---
> Ayuda: Podemos gestionar nosotros mismos los parámetros, pero cuando tenemos combinaciones complicadas es conveniente usar `CommandLineParser`, ya que [esta función](http://docs.opencv.org/3.0-beta/modules/core/doc/command_line_parser.html) lo hace por nosotros. Usa este código como base para tu programa:
--->
<!---
WM: removed bs after "original"
WM: removed bs after "printmessage"
WM: Added bs after contraste}
WM: imageFilename= 
WM: aqui -> a continuacion
WM: double -> float
--->
<!---
```cpp
const string keys =
       "{help h usage ? |   | imprimir este mensaje }"
       "{@image         |   | imagen original       }"
       "{@alpha         |   | valor de contraste    }" 
       "{@beta          |   | valor de brillo       }";

int main(int argc, char* argv[])
{
   CommandLineParser parser(argc, argv, keys);

   if (parser.has("help")) {
       parser.printMessage();
       return 0;
   }

   if (!parser.has("@image") || !parser.has("@alpha") || !parser.has("@beta")) {
       parser.printMessage();
       return -1;
   }

   string imageFilename= parser.get<string>("@image");
   double alpha = parser.get<float>("@alpha");
   double beta = parser.get<int>("@beta");

   // El codigo del ejercicio debe ir a continuacion

}
```
--->

<!---
// PROBAR: Check if params are correctly parsed in his variables -> No funciona
if (!parser.check())
{
        parser.printErrors();
        return 0;
}
-->

---

## Transformaciones globales

Una de las transformaciones globales más usadas en imagen es la transformada de Fourier. En OpenCV tenemos la función `dft` que calcula esta transformada, aunque necesitamos hacer un preproceso para preparar la entrada a esta función, y un postproceso para calcular la magnitud y la fase a partir de su resultado. En Visión por Computador no entraremos en detalles sobre cómo usar la transformada de Fourier en OpenCV, pero si quieres saber más puedes consultar [este enlace](http://docs.opencv.org/trunk/d8/d01/tutorial_discrete_fourier_transform.html).

## Transformaciones geométricas

En OpenCV la mayoría de transformaciones geométricas se implementan creando una matriz de transformación y aplicándola a la imagen original con `warpAffine`.

Esta función requiere como entrada una matriz de tamaño 2x3, ya que implementa las transformaciones afines mediante matrices aumentadas. Como hemos visto en teoría, la última fila de la matriz aumentada en una transformación afín es siempre (0,0,1) por lo que no hay que indicarla (por este motivo se indica una matriz de 2x3 en lugar de 3x3).

Veamos cómo se implementan algunas transformaciones afines con esta función.

* **Traslación**

![Matriz de traslación](images/transformaciones/translation.png)

<!---
WM: removed bs after filas
--->

<!---
M=\begin{bmatrix}
1 & 0 & t_x \\
0 & 1 & t_y \\
0 & 0 & 1 \\
\end{bmatrix}
--->

```cpp
Mat translate(Mat &src, int offsetx, int offsety) 
{
    // Creamos la matriz de traslacion (offsetx se refiere a las columnas, offsety a las filas)
    Mat M = (Mat_<float>(2,3) << 1, 0, offsetx,
                                 0, 1, offsety);
    Mat dst;
    warpAffine(src, dst, M, src.size());
    return dst;
}

Mat dst = translate(image, 0, -10);
```

La función `warpAffine` tiene también parámetros para indicar el tipo de interpolación (`flags`) y el comportamiento en los bordes, tal como puede verse en su [documentación](https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#void%20warpAffine(InputArray%20src,%20OutputArray%20dst,%20InputArray%20M,%20Size%20dsize,%20int%20flags,%20int%20borderMode,%20const%20Scalar&%20borderValue)).

En general, podemos usar `warpAffine` para implementar cualquier transformación afín, pero existen funciones específicas para ayudar a gestionar algunas transformaciones como veremos a continuación.

* **Rotación**

<!---
M=\begin{bmatrix}
cos\theta & -sin\theta & 0 \\
sin\theta & \cos\theta & 0 \\
0 & 0 & 1 \\
\end{bmatrix}
--->

<!---
Idea ejercicio próximo curso. Implementar transformación geométrica proyectiva mediante multiplicación de matrices. Una vez calculada la transformación tendrán que copiar todos los puntos <x,y> a su nueva posición en la imagen destino teniendo en cuenta los bordes. OJO: INTERPOLACION PUEDE SER JODIDA
---->

La rotación sobre un ángulo se define del siguiente modo:

![Matriz de rotación](images/transformaciones/rotation.png)

Sin embargo, OpenCV permite rotar indicando un centro de rotación ajustable para poder usar  cualquier punto de referencia como el eje. Para esto se usa la función `getRotationMatrix2D`, que recibe como primer parámetro el eje de rotación:

```cpp
Mat rotate(Mat& src, double angle) {
    Point2f center(src.cols*0.5, src.rows*0.5);
    // Creamos la matriz de rotacion
    Mat M = getRotationMatrix2D(center, angle, 1.0); // El ultimo parametro (1.0) es la escala

    Mat dst;
    warpAffine(src, dst, M, src.size(), INTER_CUBIC); // El ultimo parametro es el metodo de interpolacion (opcional)
    return dst;
}

Mat rotated = rotate(image, 90); // Rotar 90 grados
```

* **Reflexión**

Existe una función específica (`flip`) que implementa la reflexión sin necesidad de usar `warpAffine`.

```cpp
Mat dst(src.rows, src.cols, CV_8UC3);
flip(src, dst, 1);
```

El tercer parámetro de `flip` puede ser 0 (reflexión sobre el eje x), positivo (por ejemplo, 1 es reflexión sobre el eje y), o negativo (por ejemplo, -1 es sobre los dos ejes).

* **Escalado**

El escalado también se implementa mediante una función específica llamada `resize`, que permite indicar unas dimensiones concretas o una proporción entre la imagen origen y destino.

<!---
WM: tamaños
--->

```cpp
// Especificando un tamaño determinado:
resize(src, dst, Size(20,30), CV_INTER_LINEAR); // El ultimo parametro es opcional

// Especificando una escala, por ejemplo 75% de la imagen original:
resize(src, dst, Size(), 0.75, 0.75, CV_INTER_LINEAR); // El ultimo parametro es opcional
```

* **Sesgado**

Para realizar un sesgado no hay ninguna función específica, por lo que podemos crear la matriz de transformación y después usar `warpAffine`:

```cpp
Mat M = (Mat_<float>(2, 3) <<   1, 0, 0,
                              0.5, 1, 0);

Mat dst;
warpAffine(src, dst, M , Size(src.cols,src.rows));
```

* **Proyectiva**

Como hemos visto en teoría, la transformación proyectiva no es afín, por lo que no conserva el paralelismo de las líneas de la imagen original.

Para hacer una transformación proyectiva debemos indicar una matriz de 3x3 y usar la función `warpPerspective`, por ejemplo:

```cpp
Mat M = (Mat_<float>(3, 3) <<   1, 0, 0,
                              0.5, 1, 0,
                              0.2, 0, 1);

Mat dst;
warpPerspective(src, dst, M, Size(src.cols,src.rows));
```

La lista completa de parámetros de esta función puede verse en [este enlace](https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#void%20warpPerspective(InputArray%20src,%20OutputArray%20dst,%20InputArray%20M,%20Size%20dsize,%20int%20flags,%20int%20borderMode,%20const%20Scalar&%20borderValue)).

También tenemos otra opción muy práctica para implementar una transformación de este tipo, ya que suele ser muy complicado estimar a priori los valores de la matriz para realizar una transformación concreta. Esta alternativa consiste en proporcionar dos arrays de 4 puntos (`Point2f`): El primero será de la imagen original, y el segundo contiene la proyección de esos puntos (dónde van a quedar finalmente) en la imagen destino, y usar `getPerspectiveTransform` para calcular los valores de la matriz de transformación.

```cpp
 Point2f inputQuad[4];
 Point2f outputQuad[4];

 // Asignamos valores a esos puntos
 // ...

 // Obtenemos la matriz de transformación de perspectiva
 Mat lambda = getPerspectiveTransform(inputQuad, outputQuad);

 // Aplicamos la transformacion
 warpPerspective(src, dst, lambda, dst.size());
```

---

### Ejercicio

Implementa un programa llamado `perspectiva.cpp` para corregir la perspectiva de una imagen `damas.jpg` dados estos 4 puntos de sus esquinas:

<!---
WM: Añadidos puntos después de cada comentario
--->

```cpp
278, 27  // Esquina superior izquierda.
910, 44  // Esquina superior derecha.
27, 546  // Esquina inferior izquierda.
921, 638 // Esquina inferior derecha.
```

Esta es la imagen de entrada:

![damas](images/transformaciones/damas.jpg)

El programa debe guardar la imagen resultante en un fichero llamado `damas_corrected.jpg` de **tamaño 640x640 píxeles**:

![damas](images/transformaciones/damas_corrected.jpg)

<!---
Si ves que se queda corto, que marquen los puntos con el interfaz de OpenCV
-->

## Transformaciones en entorno de vecindad

En esta sección veremos cómo implementar transformaciones en entorno de vecindad usando OpenCV, en particular convoluciones y filtros de mediana.

### Filtros de convolución

Las convoluciones se implementan con la función `filter2D`.

![OpenCV kernel](images/transformaciones/anchor.png)

Esta función recibe los siguientes parámetros:

* `src`: Imagen de entrada
* `dst`: Imagen resultante
* `ddepth`: Resolución radiométrica (_depth_) de la matriz `dst`. Un valor negativo indica que la resolución será la misma que tiene la imagen de entrada.
* `kernel`: El _kernel_ a convolucionar con la imagen.
* `anchor` (opcional): La posición de anclaje del kernel (como puede verse en la figura) relativa a su origen. El punto (-1,-1) indica el centro del kernel por defecto.
* `delta` (opcional): Un valor para añadir a cada píxel durante la convolución. Por defecto, 0.
* `borderType` (opcional): El método a seguir en los bordes de la imagen para interpolación, ya que en estos puntos el filtro se sale de la imagen. Puede ser `BORDER_REPLICATE`, `BORDER_REFLECT`, `BORDER_REFLECT_101`, `BORDER_WRAP`,  `BORDER_CONSTANT`, o `BORDER_DEFAULT` (el valor por defecto).

Ejemplos de llamadas a la función:

```cpp
filter2D(src, dst, -1, kernel); // Esta forma es la más habitual
filter2D(src, dst, -1 , kernel, Point(-1,-1)); // Ancla desplazada
```

Evidentemente hay que crear antes un _kernel_ para convolucionarlo con la imagen. Por ejemplo, podría ser el siguiente:

```cpp
Mat kernel = (Mat_<int>(5, 5)  << -1, -1, -1, -1, -1,
                                  -1, -1, -1, -1, -1,
                                  -1, -1, 24, -1, -1,
                                  -1, -1, -1, -1, -1,
                                  -1, -1, -1, -1, -1);
```

> Pregunta: ¿Qué tipo de filtro acabamos de crear?

<!---
Ejemplo de ejercicio:
Haz un programa que reciba como parámetro una imagen, la lea en escala de grises y la convolucione con un filtro gaussiano de 5x5 (https://en.wikipedia.org/wiki/Kernel_(image_processing), o con media x y desviación y (por parámetro?). Guardar en otra imagen.
-->



### Filtro de mediana

El filtro de mediana se implementa de forma muy sencilla en OpenCV:

```cpp
medianBlur(src, dst, 5);
```

El último parámetro indica el tamaño del _kernel_, que siempre será cuadrado (en este ejemplo, 5x5 píxeles).

## Transformaciones morfológicas

OpenCV proporciona una serie de funciones predefinidas para realizar transformaciones morfológicas:

### Erosión y dilatación

La sintaxis de las operaciones morfológicas básicas es sencilla:

```cpp
erode(src, dst, element);
dilate(src, dst, element);
```

Ambas operaciones necesitan un elemento estructurante. Al igual que en el caso de `filter2D` se pueden añadir opcionalmente los parámetros `anchor`, `delta` y `borderType`.

Para crear el elemento estructurante se usa la función `getStructuringElement`:

```cpp
int erosion_type = MORPH_ELLIPSE; // Forma del filtro
int erosion_size = 6;             // Tamaño del filtro (6x6)

Mat element = getStructuringElement(erosion_type,
                                    Size(erosion_size, erosion_size));
```

El elemento estructurante puede tener forma de caja (`MORPH_RECT`), de cruz (`MORPH_CROSS`) o de elipse (`MORPH_ELLIPSE`).

### Apertura, cierre y Top-Hat

El resto de funciones de transformación morfológica se implementan mediante la función `morphologyEx`:

```cpp
morphologyEx(src, dst, operation, element);
```

Esta función se invoca con los mismos parámetros que `erode` o `dilate` mas un parámetro adicional que indica el tipo de operación:

* Apertura: `MORPH_OPEN`
* Cierre: `MORPH_CLOSE`
* Gradiente: `MORPH_GRADIENT`
* White Top Hat: `MORPH_TOPHAT`
* Black Top Hat: `MORPH_BLACKHAT`

En [este enlace](http://docs.opencv.org/2.4/doc/tutorials/imgproc/opening_closing_hats/opening_closing_hats.html) puedes ver código de ejemplo para implementar un interfaz que permite probar estas operaciones modificando sus parámetros.

---

### Ejercicio

El objetivo de este ejercicio (`detectarFichas.cpp`) es generar una máscara binaria que contenga sólo las fichas del juego de damas.

#### Fichas rojas

Veamos un ejemplo de detección de las fichas rojas:

![Fichas rojas](images/transformaciones/rojas.jpg)

El programa a implementar debe cargar directamente la imagen resultante del ejercicio anterior (`damas_corrected.jpg`) y a continuación seguir los siguientes pasos:

* Realizar una umbralización quedándonos sólo con los píxeles que tengan un color dentro de un rango BGR entre (0,0,50) y (40,30,255). Podemos visualizar el resultado con `imshow`. Deberíamos tener los píxeles de las fichas rojas resaltados, aunque la detección es todavía imperfecta y existen huecos.
* Crear un elemento estructurante circular de tamaño 10x10 píxeles y aplicar un operador de cierre para perfilar mejor los contornos de las fichas y eliminar estos huecos.
* Guardar la imagen resultante en el fichero `rojas.jpg`. Debería dar el mismo resultado que se muestra en la imagen anterior.

#### Fichas blancas

* Ahora intenta resaltar sólo las fichas blancas lo mejor que puedas, guardando el resultado en el fichero `blancas.jpg`. Puedes usar filtrado de color (en cualquier espacio, como HSV) y realizar transformaciones morfológicas o de cualquier otro tipo. Probablemente no te salga demasiado bien pero es un problema mucho más complicado al confundirse el color de las fichas con el de las casillas blancas.

---
