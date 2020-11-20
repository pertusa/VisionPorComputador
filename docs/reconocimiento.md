# T7- Reconocimiento de imagen

En este tema aprenderemos a buscar imágenes similares y a reconocer la clase de una imagen.

## Búsqueda por similitud

Como hemos visto en teoría, se pueden comparar descriptores para encontrar imágenes similares.

### Descriptores binarios

En el apartado `Descriptor` del tema anterior ya hemos visto un ejemplo para hacer `matching` usando ORB y la distancia de Hamming. Revísalo antes de continuar con este tema. La técnica que habíamos usado en ese ejemplo ("fuerza bruta") funciona bien con ORB porque es un descriptor binario y la comparación es muy eficiente al ser simplemente una operación XOR.

> Los descriptores binarios se idearon para hacer `matching` y funcionan mucho peor cuando se usan en problemas de reconocimiento (clasificación), aunque a veces se emplean de esta forma por eficiencia.

<!---
HOG???
--->

### Descriptores locales basados en puntos de interés

Como hemos visto en teoría, comparar descriptores como SIFT o SURF no es rápido, sobre todo si tenemos muchas imágenes en nuestra base de datos.

Para comparar dos imágenes que tienen descriptores basados en puntos de interés (y, por tanto, un número variable de elementos por imagen) se pueden usar vecinos más cercanos aproximados (en inglés, _Approximate Nearest Neighbors_). Esta técnica consiste en construir una representación interna para evitar hacer las comparaciones de todos con todos, mirando sólo aquellos que pueden ser más similares. En OpenCV tenemos una función que hace esta tarea: `FLANN`.

Podemos ver un ejemplo completo en [este enlace](https://docs.opencv.org/2.4/doc/tutorials/features2d/feature_flann_matcher/feature_flann_matcher.html). El código que realiza la comparación de los descriptores correspondientes a cada punto de interés es el siguiente:

```cpp
FlannBasedMatcher matcher;
vector<DMatch> matches;
matcher.match(descriptors_1, descriptors_2, matches);
```

En este caso, `descriptors_1` contiene todos los descriptores de la primera imagen y `descriptors_2` los de la segunda. La función `match` construye un sistema de búsqueda de vecinos más cercanos aproximados usando estos descriptores de entrada, y luego usa estos vecinos para encontrar las mejores correspondencias entre los descriptores de ambas imágenes. En `matches` se guarda un vector de elementos `DMatch`, que es un registro que contiene:

* `distance`: La distancia entre los dos keypoints
* `queryIdx`: El índice (en el vector) del keypoint de la primera imagen
* `trainIdx`: El índice (en el vector) del keypoint de la segunda imagen

Como puedes ver en el código, se suelen eliminar los pares de puntos cuya distancia es mayor que un cierto umbral.

> Nota: El código anterior usa SURF, que está patentado, por lo que sólo podrás compilarlo y ejecutarlo si has instalado el paquete `contrib` y además has aceptado la licencia correspondiente.

## Reducción de dimensionalidad

Para hacer que el `matching` sea más eficiente (aunque normalmente a costa de empeorar un poco los resultados) podemos reducir el tamaño de los descriptores usando alguna de las siguientes técnicas vistas en teoría.

### Bag of Words (BoW)

Como hemos visto en teoría, usar BoW es mucho más eficiente que el descriptor completo cuando se trata de descriptores como SIFT o SURF (no binarios).

> Importante: La implementación de BoW en OpenCV no permite el uso de descriptores binarios, ni tampoco de  aquellos que no pertenezcan a la clase DescriptorExtractor.

<!---
https://www.codeproject.com/Articles/619039/Bag-of-Features-Descriptor-on-SIFT-Features-with-O
-->

Usando el siguiente código podemos entrenar un diccionario BoW a partir de todos los descriptores extraídos de todas las imágenes del conjunto de entrenamiento:

<!---
WM: 
tc~(CV
"dictionary.yml"~,
extraidos de todas las imagenes -> extraidos de las imagenes
--->

```cpp
// Funcion que recibe los datos de todos los descriptores extraídos de todas las imágenes (un descriptor
// por fila) en la variable allFeatures, y el numero de palabras que queremos que tenga nuestro diccionario
void trainBOW(const Mat &allFeatures, int k) 
{
   // Establecemos un criterio de finalización para k-means
   TermCriteria tc (CV_TERMCRIT_ITER, 100, 0.001);

   // Creamos un objeto BOWTrainer indicando que nuestro diccionario tendrá 100 palabras, por ejemplo.
   BOWKMeansTrainer BOWTrainer(k, tc, 1, KMEANS_PP_CENTERS);

   // Lo entrenamos con los datos de todos los descriptores extraidos de  las imagenes
   Mat dictionary = BOWTrainer.cluster(allFeatures);

   // Guardamos el diccionario en el fichero de texto dictionary.yml
   FileStorage fs("dictionary.yml" , FileStorage::WRITE);
   fs << "vocabulary" << dictionary;
   fs.release();
}
```

De esta forma hemos entrenado un diccionario de `k` palabras. A continuación podemos extraer un descriptor, y convertirlo en un histograma de palabras (este será nuestro nuevo descriptor). Para ello primero debemos inicializar nuestro descriptor BOW:

<!---
WM:
Las siguientes 4 instrucciones -> Las siguientes instrucciones
por ejemplo usando SURF -> por ejemplo, usando SURF
la funcion "compute"  -> la funcion compute

--->

```cpp
BOWImgDescriptorExtractor initBOWDescriptor() {
  // Las siguientes instrucciones son para el caso de que necesitemos cargar el diccionario almacenado
  // previamente (si ya lo tenemos en dictionary, no hacen falta)
  Mat dictionary;
  FileStorage fs("dictionary.yml", FileStorage::READ);
  fs["vocabulary"] >> dictionary;
  fs.release();

  // Inicializamos nuestro descriptor, por ejemplo, usando SURF
  Ptr<DescriptorExtractor> descriptors = DescriptorExtractor::create("SURF");
  // Indicamos el algoritmo de matching para calcular la distancia entre el descriptor y su cluster mas cercano
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");  
  // Con los elementos anteriores, inicializamos nuestro descriptor BOW
  BOWImgDescriptorExtractor BowDE(extractor, matcher);
  // Le asignamos el vocabulario
  BowDE.setVocabulary(dictionary);

  return BowDe;
}
```  

Y ahora ya podemos aplicarlo sobre una imagen:

```cpp
Mat extractBOWDescriptor(const Mat &image, BOWImgDescriptorExtractor &BowDE)
{
  // Detectamos los keypoints
  Ptr<FeatureDetector> detector = FeatureDetector::create("SURF");
  vector<KeyPoint> keypoints;
  detector->detect(image, keypoints);

  // Calculamos el descriptor BOW (histograma de palabras) para una imagen dados todos sus keypoints.
  // Para esto, la funcion compute extrae el descriptor SURF y despues crea el histograma de palabras
  Mat BowDescriptor;
  BowDE.compute(image, keypoints, BowDescriptor);

  return BowDescriptor;
}
```

No haremos ningún ejercicio con BoW porque sólo se puede usar con descriptores patentados, pero esta información puede ser útil para tu proyecto.

### PCA y LDA

Podemos ver un ejemplo completo de reducción de dimensionalidad mediante PCA en [este enlace](https://bytefish.de/blog/pca_in_opencv/). La parte interesante está en el `main`:

```cpp
// Calculamos PCA a partir de data, los primeros num_components como maximo
// Los eigenvectors de PCA se guardan en el vector pca.eigenvectors
PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, num_components);
```

Como ves, PCA es un algoritmo de aprendizaje no supervisado, es decir, para calcularlo no se necesitan las muestras etiquetadas. En OpenCV podemos usar también LDA. El algoritmo LDA era originalmente no supervisado, pero posteriormente surgieron variantes que aceptan etiquetas.

<!---
http://answers.opencv.org/question/64165/how-to-perform-linear-discriminant-analysis-with-opencv/
-->

Tampoco entraremos en detalles sobre estas técnicas, ya que las veréis en una asignatura del grado, pero podéis usarlas para extraer descriptores más compactos en vuestro proyecto.

## Detección de caras

<!---
https://github.com/opencv/opencv/blob/master/samples/cpp/facedetect.cpp

https://github.com/opencv/opencv/blob/master/samples/cpp/dbt_face_detection.cpp
-->

<!---
Ojo, faceRecognizer está en contrib!: https://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html
-->

A continuación puedes ver un ejemplo de un programa completo en OpenCV para detectar caras completas y ojos en vídeos. Se trata de una versión ligeramente modificada del código de ejemplo de OpenCV que puedes ver en [este](https://docs.opencv.org/3.3.0/db/d28/tutorial_cascade_classifier.html) enlace.

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */
String face_cascade_name, eyes_cascade_name;
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
String window_name = "Capture - Face detection";

/** @function main */
int main( int argc, const char** argv )
{
    CommandLineParser parser(argc, argv,
        "{help h||}"
        "{face_cascade|haarcascade_frontalface_alt.xml|}"
        "{eyes_cascade|haarcascade_eye_tree_eyeglasses.xml|}");

    cout << "\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n";
    parser.printMessage();

    face_cascade_name = parser.get<string>("face_cascade");
    eyes_cascade_name = parser.get<string>("eyes_cascade");
    VideoCapture capture;
    Mat frame;

    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };

    //-- 2. Read the video stream
    capture.open( 0 );
    if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return -1; }

    while ( capture.read(frame) )
    {
        if( frame.empty() )
        {
            printf(" --(!) No captured frame -- Break!");
            break;
        }

        //-- 3. Apply the classifier to the frame
        detectAndDisplay( frame );

        char c = (char)waitKey(10);
        if( c == 27 ) { break; } // escape
    }
    return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
    std::vector<Rect> faces;
    Mat frame_gray;

    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

        Mat faceROI = frame_gray( faces[i] );
        std::vector<Rect> eyes;

        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );

        for ( size_t j = 0; j < eyes.size(); j++ )
        {
            Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
        }
    }
    //-- Show what you got
    imshow( window_name, frame );
}
```

Prueba este código descargando los que contienen los [modelos entrenados](https://github.com/opencv/opencv/tree/master/data/haarcascades) con descriptores Haar. También puedes usar descriptores LBP descargando sus correspondientes modelos desde [este otro enlace](https://github.com/opencv/opencv/tree/master/data/lbpcascades). Si tu ordenador no tiene webcam, prueba a usar un video cualquiera de internet en el que aparezcan caras.

Como puedes ver en los modelos Haar disponibles en el enlace anterior, es posible usar este mismo código para detectar matrículas, peatones, gatos, etc. simplemente cambiando el modelo.

Puedes usar cualquier tipo de imágenes para entrenar tu propio modelo usando el siguiente programa, que se instala con OpenCV:

```bash
opencv_traincascade
```

La ayuda completa para usar este programa está [en este enlace](https://docs.opencv.org/3.0-beta/doc/user_guide/ug_traincascade.html?highlight=train%20cascade), pero en [aquí](https://github.com/spmallick/opencv-haar-classifier-training) puedes ver un ejemplo completo de uso que también construye los datos de entrenamiento a partir de las imágenes. El primer paso para ejecutar `opencv_traincascade` es crear un conjunto de entrenamiento con muestras positivas (por ejemplo, caras) y negativas (no caras). Después, se entrena un clasificador de tipo `Boosting` (por defecto, `AdaBoost`) en cascada usando descriptores Haar o LBP mediante una instrucción como esta:

```bash
opencv_traincascade -data classifier -vec samples.vec -bg negatives.txt\
  -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 1000\
  -numNeg 600 -w 80 -h 40 -mode ALL -precalcValBufSize 1024\
  -precalcIdxBufSize 1024
```

<!---
## Reconocimiento de caras?
-->

## Clasificación

Como hemos visto en teoría y al principio de este capítulo, una forma fácil de clasificar una imagen es mediante búsqueda de imágenes similares que estén etiquetadas, devolviendo la clase de la imagen más similar de nuestra base de datos.

--------

### Ejercicio

Vamos a hacer un ejercicio en el que extraeremos un descriptor ORB de una imagen y lo compararemos con los de otras imágenes ya etiquetadas para obtener su clase.

Para esto tenemos que [descargar](http://www.dlsi.ua.es/~pertusa/mirbot-exercises.zip) un subconjunto de imágenes etiquetadas de la base de datos MirBot. [MirBot](http://www.mirbot.com) es un proyecto desarrollado en la UA para hacer un sistema de reconocimiento interactivo de imágenes para móviles. Cuanto más usuarios tiene y más fotos se añaden, mejor funciona el sistema.

Para este caso sólo vamos a usar un subconjunto de las imágenes enviadas por los usuarios, en concreto algunas pertenecientes a estas 10 clases: _book, cat, cellphone, chair, dog, glass, laptop, pen, remote, tv_. Descomprime el fichero descargado y echa un vistazo para ver los casos que intentamos reconocer.

Ahora se trata de completar los puntos marcados con `TODO` en el siguiente código para realizar la clasificación. Guárdalo con el nombre `orbBF.cpp` e intenta entender bien todas las instrucciones antes de empezar a modificarlo.

<!---
WM: 
labelNames = -> labelNames= 
imagenes -> imágenes
parametro -> parámetro
fi >> imagefn >> label -> fi>>imagefn>>label
Funcion -> Función
Extrayendo descriptores... -> Quitados los 3 puntos
(float)(test.size())
--->


```cpp
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>

using namespace cv;
using namespace std;

const vector<string> labelNames= {"book", "cat", "chair", "dog", "glass", "laptop", "pen", "remote", "cellphone", "tv"};

// Leemos las imágenes especificadas en un fichero de texto y las guarda en images, junto con sus etiquetas en labels.
// El ultimo parámetro (max) podemos indicarlo cuando queramos que no se cargen todos los datos sino como maximo max imagenes (para hacer pruebas mas rapido)
void readData(string filename, vector<Mat> &images, vector<int> &labels, int max=-1 )
{
   ifstream fi(filename.c_str());

   if (!fi.is_open()) {
     cerr << "Fichero " << filename << " no encontrado" << endl;
     exit(-1);
   }

   cout << "Cargando fichero " << filename << "..." << endl;

   string imagefn;
   string label;

   for (int i=0; i!=max && fi>>imagefn>>label; i++) {
       Mat image = imread(imagefn, IMREAD_GRAYSCALE);
       int label_u = distance(labelNames.begin(), find(labelNames.begin(), labelNames.end(), label));

       images.push_back(image);
       labels.push_back(label_u);
   }
   fi.close();
}

// Función para extraer las caracteristicas ORB de una imagen
vector<Mat> extractORBFeatures(const vector<Mat> &data)
{
     vector<Mat> features;

     // TODO: Recorremos el vector data, y para cada imagen (Mat) detectamos y extraemos caracteristicas ORB con 100 puntos como máximo por imagen.
     // Al final debemos añadir todos todos los descriptores obtenidos de una imagen en un elemento del vector "features" (usando push_back).


     return features;
}

// Extraemos y guardamos todas las caracteristicas en el fichero trainORBDescriptors.yml, para poder usarlas luego en la fase de reconocimiento
void storeTrainORBDescriptors(const vector<Mat> &train, const vector<int> &labelsTrain) {

        cout << "Extrayendo descriptores" << endl;
        vector<Mat> features = extractORBFeatures(train);

        cout << "Guardando los descriptores en el fichero trainORBDescriptors.yml" << endl;

        FileStorage fs("trainORBDescriptors.yml", FileStorage::WRITE);
        fs << "ORBFeatures" << "[";
        for (unsigned i=0; i<train.size(); i++)  {
           fs << "{" << "label" << labelsTrain[i] << "descriptor" << features[i] << "}";
        }
        fs << "]";

        fs.release();
}

// Esta es la fase de reconocimiento de imagenes no vistas durante el entrenamiento. Para esto usamos el conjunto de test.
void testORB(const vector<Mat> &test, const vector<int> &labelsTest)
{
       // Cargar fichero trainORBDescriptors.yml, guardando en trainORBDescriptors los descriptores de las imagenes de entrenamiento y en labelsTrain sus etiquetas.
       cout << "Cargando descriptores de entrenamiento" << endl;
       FileStorage fs("trainORBDescriptors.yml", FileStorage::READ);

       vector<Mat> trainORBDescriptors;
       vector<int> labelsTrain;

       FileNode features = fs["ORBFeatures"];
       for (FileNodeIterator it = features.begin(); it!= features.end(); it++) {
           int label = (*it)["label"];
           Mat desc;
           (*it)["descriptor"]>>desc;
           trainORBDescriptors.push_back(desc);
           labelsTrain.push_back(label);
       }

       // Calculamos descriptores ORB del conjunto de test
       cout << "Calculando descriptores de test" <<  endl;
       vector<Mat> testORBDescriptors = extractORBFeatures(test);

       // Comparamos cada imagen de test con todas las de train para obtener su etiqueta. Ojo, a pesar de ser un descriptor binario veras que es lento.
       cout << "Matching..." << endl;
       vector<DMatch> matches;
       BFMatcher matcher(NORM_HAMMING);


       // TODO: Completar los siguientes bucles para comparar los descriptores de cada imagen de test con todas las imagenes de entrenamiento.
       int ok=0;
       int predicted;
       for (unsigned i = 0; i<testORBDescriptors.size(); i++) {
         for (unsigned j=0; j<trainORBDescriptors.size(); j++) {
            if (testORBDescriptors[i].cols!=0 && trainORBDescriptors[j].cols!=0) {  // Solo si el descriptor no es vacio

               // TODO: Solo consideramos que dos puntos son similares si su distancia es menor de 90. 
               // La imagen mas similar sera la que tiene mas keypoints coincidentes. Hay que extraer su etiqueta y guardarla en "predicted".

            }
         }

         cout << i << endl;
         // Si la etiqueta de la imagen de test coincide con la que deberia dar, entonces la prediccion se considera correcta..
         if (predicted == labelsTest[i])
           ok++;
       }

       float accuracy = (float)ok/(float)(test.size());
       cout << "Accuracy=" << accuracy << endl;
}

int main(int argc, char *argv[])
{
    if (argc!=2 || (string(argv[1])!="train" && string(argv[1])!="test")) {
       cout << "Sintaxis: " << argv[0] << " <train/test>" << endl;
       exit(-1);
    }

    vector<Mat> train, test;
    vector<int> labelsTrain, labelsTest;

    if (string(argv[1]) == "train") {
        readData("train.txt", train, labelsTrain);
        storeTrainORBDescriptors(train, labelsTrain);
    }

    else {
        readData("test.txt", test, labelsTest);
        testORB(test, labelsTest);
    }
}
```

Debes compilar el programa con la opción `std=c++11` de esta forma:

```bash
g++ -o orbBF orbBF.cpp `pkg-config --cflags --libs opencv` --std=c++11
```

Para ejecutarlo, debes indicar la opción "train" o "test". Ejemplo:

```bash
./orbBF train
```

Una vez completes el código, la primera fase es ejecutarlo con `train` para extraer el fichero de características ORB de todas las imágenes de entrenamiento. Después se puede ejecutar en modo `test` para reconocer todas las imágenes del conjunto de test y compararlas con sus etiquetas reales.

El resultado final tras la fase de test debería ser parecido a este:

```bash
Accuracy=0.194
```

Puede haber pequeñas variaciones porque hay una inicialización aleatoria en el entrenamiento, pero el resultado debe ser similar al anterior.

--------

### Ejercicio

Con el ejercicio anterior hemos hecho una primera aproximación para reconocer objetos pero, como hemos visto en teoría, es mucho más eficiente entrenar un clasificador específico para nuestros datos. De esta forma no es necesario consultar todas las imágenes de entrenamiento para buscar la imagen más cercana como se hace con kNN.

En la asignatura no hemos visto cómo funcionan internamente los algoritmos de aprendizaje supervisado, pero vamos a usar uno en modo "caja negra" para mejorar el porcentaje de acierto que hemos obtenido en el ejercicio anterior.

Partimos de este otro código, que como verás tiene partes comunes con el anterior. Guárdalo con el nombre `HOGSVM.cpp` y complétalo con los comentarios marcados con `TODO`:

<!---
WM: 
labelNames = -> labelNames=
tamanyo -> tamaño
Anyadimos -> Añadimos
j<descriptor_size -> j< descriptor_size
EPS=1e-6 -> EPS=1e-5 (sale igual)

".push_back(image); " -> ".push_back(image);
imagefn,~IMREAD_GRAYSCALE
escalar a 128x128 
svm = SVM::create();
 -> svm= SVM::create(); 
 --->


```cpp
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>

using namespace cv;
using namespace std;
using namespace cv::ml;

const vector<string> labelNames={"book", "cat", "chair", "dog", "glass", "laptop", "pen", "remote", "cellphone", "tv"};

// Lee las imagenes especificadas en un fichero de texto y las guarda en images, junto con sus etiquetas en labels.
// El ultimo parametro (max) podemos indicarlo cuando queramos que no se cargen todos los datos sino solo hasta max imagenes (para hacer pruebas mas rapido)
void readData(string filename, vector<Mat> &images, vector<int> &labels, int max=-1)
{
   ifstream fi(filename.c_str());

   if (!fi.is_open()) {
     cerr << "Fichero " << filename << " no encontrado" << endl;
     exit(-1);
   }

   cout << "Cargando " << filename << "..." << endl;

   string imagefn;
   string label;

   for (int i=0; i!=max && fi >> imagefn >> label; i++) {
       Mat image = imread(imagefn, IMREAD_GRAYSCALE);
       int label_u = distance(labelNames.begin(), find(labelNames.begin(), labelNames.end(), label));

       images.push_back(image);
       labels.push_back(label_u);
   }
   fi.close();
}

// Extraemos los descriptores HOG (tendremos un vector<float> por cada imagen) del conjunto recibido por parametro
vector<vector<float> > extractHOGFeatures(const vector<Mat> &data)
{
     vector<vector<float> > features;
     HOGDescriptor hog;

     for (unsigned i = 0; i<data.size(); i++)  {
           Mat image = data[i];

           // TODO: Para que todas las imagenes tengan el mismo tamaño de descriptor, las debemos escalar a 128x128
           // TODO: Ahora debemos calcular el descriptor HOG con un stride de 128x128 (asumimos que el objeto ocupa toda la imagen) y un padding (0,0).
           // TODO: Añadimos el descriptor obtenido de la imagen al vector features con push_back.

     }
     return features;
}

// Funcion para convertir un vector<vector<float> > a una matriz Mat, ya que nos hace falta este formato para nuestro clasificador
Mat convertToMat(vector<vector<float> > features)
{
    int descriptor_size = features[0].size();
    Mat trainMat(features.size(), descriptor_size, CV_32F);
    for(int i=0; i<features.size(); i++){
        for(int j = 0; j< descriptor_size; j++){
           trainMat.at<float>(i,j) = features[i][j];
        }
    }
    return trainMat;
}

// Entrenamiento de nuestro clasificador, en este caso un SVM (support vector machine).
void trainHOGSVM(const vector<Mat> &train, const vector<int> &labelsTrain) {

    cout << "Extrayendo descriptores..." << endl;
    vector<vector<float> > features = extractHOGFeatures(train);

    // Convertimos a Mat
    Mat trainMat = convertToMat(features);
    Mat dataLabel(labelsTrain);

    // Configuramos el clasificador SVM
    cout << "Entrenando..." << endl;
    Ptr<SVM> svm= SVM::create();
    // TODO: Debemos poner el clasificador con el tipo C_SVC
    // TODO: Su kernel debe ser LINEAR
    // TODO: El criterio de finalización debe ser MAX_ITER con 100 iteraciones maximas y EPS=1e-5.
    // Ayuda para los puntos anteriores: https://docs.opencv.org/3.3.0/d1/d2d/classcv_1_1ml_1_1SVM.html


    // Entrenamos SVM con trainMat y dataLabel
    Ptr<TrainData> data = TrainData::create(trainMat, ROW_SAMPLE, dataLabel);
    svm->train(data);

    // Guardamos el modelo en un fichero.
    svm->save("modelSVM.xml");
}

// Probamos el modelo SVM con el conjunto de test
void testHOGSVM(const vector<Mat> &test, const vector<int> &labelsTest)
{
    cout << "Extrayendo descriptores..." << endl;
    vector<vector<float> > features = extractHOGFeatures(test);
    Mat testMat = convertToMat(features);
    Mat dataLabel(labelsTest);

    // Cargamos modelo
    cout << "Cargando modelo..." << endl;
    Ptr<SVM> svm = Algorithm::load<SVM>("modelSVM.xml");

    // Hacemos el reconocimiento
    cout << "Clasificando muestras..." << endl;
    Mat testResponse;
    svm->predict(testMat, testResponse);

    // Calculamos accuracy
    int ok=0;
    for(int i=0; i<testResponse.rows; i++) {
        int predicted = testResponse.at<float>(i,0);
        if (predicted==labelsTest[i]){
            ok++;
        }
    }
    float accuracy = ((float)ok/testResponse.rows);
    cout << "Accuracy=" << accuracy << endl;
}


int main(int argc, char *argv[])
{
    if (argc!=2 || (string(argv[1])!="train" && string(argv[1])!="test")) {
       cout << "Sintaxis: " << argv[0] << " <train/test>" << endl;
       exit(-1);
    }

    vector<Mat> train, test;
    vector<int> labelsTrain, labelsTest;

    if (string(argv[1]) == "train") {
        readData("train.txt", train, labelsTrain);
        trainHOGSVM(train, labelsTrain);
    }
    else {
        readData("test.txt", test, labelsTest);
        testHOGSVM(test, labelsTest);
    }
}
```

En este caso se trata de extraer los descriptores HOG de las imágenes del conjunto `train` para entrenar un clasificador supervisado de tipo _Support Vector Machine_ (SVM). Una vez entrenado el modelo, se guardará en el fichero `modelSVM.xml`. De esta forma, en la la fase de reconocimiento (`test`) se cargará el modelo para predecir la clase de una imagen desconocida. En este programa probaremos a reconocer todas las del conjunto test, al igual que en el anterior.

El resultado tras comprobar el conjunto de test debe ser similar a este:

```bash
Accuracy=0.477
```

Como ves, esta técnica, además de ser mucho más rápida, mejora claramente el resultado respecto al  ejercicio anterior.

<!---
## TrainSVM
https://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial/
--->
