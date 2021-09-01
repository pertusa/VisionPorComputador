# Instalación de OpenCV3

[OpenCV](http://opencv.org) es la librería más usada para procesamiento de imágenes, y puede usarse en Windows, Linux, MacOS, iOS y Android. También puede [integrarse con ROS](http://wiki.ros.org/vision_opencv), lo que hace que OpenCV sea muy utilizado en el ámbito de la robótica. Está disponible para Python y C++.

En la asignatura de Visión por Computador usaremos **OpenCV 4.5 sobre Linux en Python3**. Para hacer las prácticas también puedes instalarlo en MacOS o en Windows.

## Instalación en Windows, MacOs y Linux

Hay dos opciones para instalar todo el software de la asignatura:

### Instalación con miniconda (recomendado)

Este tipo de instalación hace que las librerías de python necesarias para la asignatura no interfieran con otras versiones de librerías que ya tengáis instaladas en el sistema.

Para esto, debéis instalar miniconda usando [este enlace](https://docs.conda.io/en/latest/miniconda.html). 

Una vez instalado, ejecutar desde el terminal (sólo la primera vez):

```bash
 conda create -n vision
```

Esto crea un **entorno de python**. Cada vez que se inicie un nuevo terminal es necesario activar este entorno:

```bash
conda activate vision
```

Una vez dentro del entorno se pueden instalar librerías de python (esto sólo hay que hacerlo una vez, ya que quedan instaladas para dicho entorno) o incluso paquetes de linux con `apt-get`, y estos se instalarán solo para el entorno:

```bash
pip3 install opencv-contrib-python numpy matplotlib pandas scikit-image scikit-learn
```

Se puede salir del entorno con el siguiente comando:

```bash
conda deactivate
```

### Instalación directa

Desde el terminal (con python3 previamente instalado), ejecutar directamente:

```bash
pip3 install opencv-contrib-python numpy matplotlib pandas scikit-image scikit-learn
````

La ventaja es que no tendremos que cambiar al entorno cada vez que se arranca un terminal, pero el inconveniente de que si tenéis otras asignaturas (o en general otro software) que necesite versiones distintas de alguna de estas librerías puede haber problemas de compatibilidad.
