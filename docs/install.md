# Instalación de OpenCV

[OpenCV](http://opencv.org) es la librería más usada para procesamiento de imágenes, y puede instalarse en Windows, Linux, MacOS, iOS y Android. También puede [integrarse con ROS](http://wiki.ros.org/vision_opencv), lo que hace que OpenCV sea muy utilizado en el ámbito de la robótica. Está disponible para C++, Python y Java.

En la asignatura de Visión por Computador usaremos **OpenCV 4.5 en Python3**. Para hacer las prácticas en los laboratorios trabajaremos en Linux (recomendado), aunque también puedes hacer los ejercicios en MacOS o en Windows.

Hay dos opciones para instalar el software necesario para la asignatura, tanto en Linux como en MacOS o Windows: miniconda (recomendado) o instalación directa.

### Instalación con miniconda (recomendado)

Este tipo de instalación hace que las librerías de python necesarias para la asignatura no interfieran con las versiones de otras librerías que ya tengáis instaladas en el sistema.

Para esta opción debéis instalar miniconda usando [este enlace](https://docs.conda.io/en/latest/miniconda.html).

> **Importante**: Descargad la versión correspondiente a vuestro sistema operativo que ponga **python 3.9**. 

Si usáis Linux, desde un terminal hay que dar permisos de ejecución al fichero descargado. Por ejemplo:

```zsh
chmod +x ./Miniconda3-py39_4.12.0-Linux-x86_64.sh
```

Una vez tenemos los permisos podemos instalar conda:
```zsh
./Miniconda3-py39_4.12.0-Linux-x86_64.sh
```

Y una vez instalado, debemos ejecutar (sólo una vez) desde el terminal:

```zsh
 conda create -n vision python=3
```

Esto crea un **entorno de python3**. Cada vez que se inicie un nuevo terminal para hacer código de Visión por Computador tendrás que activar el entorno:

```zsh
conda activate vision
```

Una vez dentro del entorno se pueden instalar librerías de python (esto sólo hay que hacerlo una vez, ya que quedan instaladas para dicho entorno) o también paquetes de linux con `apt-get`. Estos se instalarán solo para el entorno:

```zsh
pip3 install opencv-contrib-python numpy matplotlib pandas scikit-image scikit-learn
```

Se puede salir del entorno con el siguiente comando:

```zsh
conda deactivate
```

### Instalación directa

Desde el terminal (con python3 previamente instalado) se puede ejecutar directamente:

```zsh
pip3 install opencv-contrib-python numpy matplotlib pandas scikit-image scikit-learn
```

La ventaja es que no tendremos que cambiar de entorno cada vez que abrimos un nuevo terminal, pero el inconveniente de que si tenéis otras asignaturas (o en general otro software) que necesite versiones distintas de alguna de estas librerías puede haber problemas de compatibilidad.

## Edición de código en python

Se puede usar cualquier editor para python, pero recomendamos instalar [Visual Studio Code](https://code.visualstudio.com). Este software está disponible para Linux, Mac y Windows.
