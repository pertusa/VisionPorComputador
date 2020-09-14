# Instalación de OpenCV3

[OpenCV](http://opencv.org) es la librería más usada para procesamiento de imágenes, y puede usarse en Windows, Linux, MacOS, iOS y Android. También puede [integrarse con ROS](http://wiki.ros.org/vision_opencv), lo que hace que OpenCV sea muy utilizado en el ámbito de la robótica. Está disponible para Python y C++.

En la asignatura de Visión por Computador usaremos **OpenCV 3.4.6 sobre Linux en C++**. Para hacer las prácticas puedes también instalarlo en MacOS, pero se desaconseja usar Windows.

En esta parte veremos cómo instalar OpenCV 3.4.6. Para empezar debemos descargar el código fuente (en `.zip`) desde [aquí](https://github.com/opencv/opencv/releases/tag/3.4.6).

## Instalación en Linux (Ubuntu 16.04 y Ubuntu 20.04)

Si queremos instalarlo en **Ubuntu 16.04** o **Ubuntu 20.04** debemos descomprir el código fuente y compilarlo siguiendo los pasos que pueden verse en [esta url](http://www.codebind.com/cpp-tutorial/install-opencv-ubuntu-cpp/).

## Instalación en Linux (Ubuntu 18.04)

Instalarlo en **Ubuntu 18.04** es mucho más sencillo, simplemente hay que indicar lo siguiente desde el terminal:

```bash
sudo apt-get install libopencv-dev
```

Esta instrucción instalará OpenCV 3.2, que también puedes usar en la asignatura.

<!---
1. Instalar los paquetes de los que depende OpenCV3:
```bash
sudo apt-get install build-essential
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
sudo apt-get install libqt4-dev-bin libqt4-help libqt4-scripttools libqt4-test qt4-qmake libqt4-dev libqt4-opengl-dev
```
2. Preparar la compilación de OpenCV3:
```bash
cd ~/opencv-3.4.3
mkdir release
cd release
cmake -D BUILD_TIFF=ON CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local \
-D WITH_TBB=ON -D WITH_V4L=ON \
-D INSTALL_C_EXAMPLES=ON -D BUILD_EXAMPLES=ON \
-D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules /opt/opencv/ \
-D WITH_QT=ON -D WITH_GTK=ON -D WITH_OPENGL=ON ..
```

3. Compilar OpenCV3. Desde el directorio `release` ejecutamos:
```bash
make -j2
sudo make install
```
4. Añadir esta línea al final del fichero `.profile` que ya existe en el directorio principal de tu usuario (está oculto, para verlo desde el terminal debes indicar `ls -l`). Para esto puedes usar cualquier editor de texto, como por ejemplo `gedit`:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/opencv/lib
```
Por último cerramos la terminal y volvemos a abrirla para que los cambios de `.profile` sean efectivos.

-->

## Instalación de la máquina virtual de la EPS

Alternativamente puedes instalar [VirtualBox](https://www.virtualbox.org) en tu ordenador, y ejecutar la máquina virtual que contiene el software de los laboratorios.  En [este enlace](https://blogs.ua.es/labseps/2019/11/05/virtual-ubuntu-eps-2019-vdi/) puedes encontrar las instrucciones de instalación de este VDI que han creado los técnicos de la EPS para el curso 2019/2020. El VDI del curso 2020/2021 se publicará durante el curso, pero en el caso de Visión por Computador el software a utilizar no cambia con respecto al curso 2019/2020.

## Instalación en Microsoft con Ubuntu Terminal

Las instrucciones de instalación son las mismas que la versión de Ubuntu que lleve equipada Windows. Por ejemplo, si tiene instalada la 20.04 hay que descargar el código y compilarlo, si la versión es 18.04 hay que usar simplemente el instalador de paquetes de Ubuntu.

## Instalación en MacOS

Si queremos instalarlo en **MacOS** se recomienda usar `homebrew`. Para
esto, debemos:

* Instalar XCode desde App Store (para el compilador de g++)
* Instalar homebrew siquiendo estas [instrucciones](https://brew.sh).
* Desde el terminal, ejecutar:

```bash
brew update
brew install opencv@3
brew upgrade
```

Añadir las siguientes líneas al fichero .bash_profile que está en el
directorio raiz del usuario (si no existe, debes crearlo):

```bash
export LDFLAGS="-L/usr/local/Cellar/opencv\@3/3.4.5_6/lib"
export CPPFLAGS="-I/usr/local/Cellar/opencv\@3/3.4.5_6/include"
export PKG_CONFIG_PATH="/usr/local/Cellar/opencv\@3/3.4.5_6/lib/pkgconfig"
```

Si la versión instalada es distinta de la 3.4.5_6, debes cambiar el número en las
líneas anteriores.

Por último, debemos ejecutar desde el terminal:

```bash
cp /usr/local/Cellar/opencv@3/3.4.5_6/lib/pkgconfig/opencv.pc /usr/local/lib/pkgconfig/
pkg-config update
```

Hay que cerrar el terminal y volver a abrirlo para probar que se puede compilar
usando opencv.
 
Si tienes MacPorts en lugar de HomeBrew, entonces es mejor seguir [este otro](http://tilomitra.com/opencv-on-mac-osx/) tutorial.
