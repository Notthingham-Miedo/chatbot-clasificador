# 🧠 Clasificador de Reseñas - Sistemas Informáticos

Una aplicación web desarrollada con Flask que utiliza redes neuronales para clasificar reseñas de sistemas informáticos en sentimientos: **Positivo**, **Negativo** o **Neutral**. Para esto, se utilizó el mismo entrenamiento de red neuronal utilizando en la actividad práctica evaluativa N° 2.

## 👥 Integrantes del Proyecto
- Botta, Francisco: bottafrancisco01@gmail.com
- Carnino, Martin: tinchoogd01@gmail.com
- Griffone, Bruno: brunogriffone15@gmail.com 
- Gimenez, Tomás: tomasgimenez7.tg@gmail.com
- Sanchez, Facundo: facusanchez105@gmail.com

## 📋 Descripción del Proyecto

Este proyecto académico implementa un clasificador de sentimientos usando:
- **Red Neuronal Multicapa (MLP)** con Keras/TensorFlow
- **Vectorización TF-IDF** para procesamiento de texto
- **Interfaz web moderna** con Flask y Bootstrap 5
- **API RESTful** para clasificación en tiempo real

## 🎯 Características

- Interfaz web intuitiva y responsiva
- Clasificación de sentimientos en tiempo real
- Indicador de confianza del modelo
- Ejemplos interactivos predefinidos
- Sistema de validación y manejo de errores
- Diseño moderno con efectos visuales

## 📁 Estructura del Proyecto

```
flask-classifier/
├── venv/                    # Entorno virtual (se crea automáticamente)
├── app.py                   # Aplicación Flask principal
├── train_model.py           # Script para entrenar el modelo
├── models/                  # Directorio para modelos entrenados
│   ├── modelo_clasificador.h5
│   ├── vectorizer.pkl
│   └── label_encoder.pkl
├── templates/
│   └── index.html          # Interfaz web
├── static/
│   └── style.css           # Estilos personalizados
├── dataset_resenas.csv     # Dataset de entrenamiento
├── requirements.txt        # Dependencias del proyecto
└── README.md              # Este archivo
```

## 🚀 Instalación y Configuración

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Paso 1: Clonar o Crear el Proyecto

```bash
# Crear directorio del proyecto
mkdir flask-classifier
cd flask-classifier
```

### Paso 2: Crear Entorno Virtual

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate

# En macOS/Linux:
source venv/bin/activate
```

### Paso 3: Instalar Dependencias

```bash
# Instalar todas las dependencias necesarias
pip install flask tensorflow scikit-learn pandas numpy matplotlib joblib

# Generar archivo de dependencias
pip freeze > requirements.txt
```

**Alternativamente**, si ya tienes el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Paso 4: Crear Estructura de Carpetas

```bash
# Crear carpetas necesarias
mkdir templates
mkdir static
mkdir models
```

### Paso 5: Agregar los Archivos del Proyecto

Copia todos los archivos proporcionados en sus respectivas carpetas:
- `app.py` en la raíz del proyecto
- `train_model.py` en la raíz del proyecto
- `index.html` en la carpeta `templates/`
- `style.css` en la carpeta `static/`

## 🎯 Ejecución del Proyecto

### Paso 1: Entrenar el Modelo

```bash
# Entrenar y guardar el modelo
python train_model.py
```

**Salida esperada:**
```
Archivo CSV no encontrado. Creando dataset demo...
Dataset demo creado y guardado
Entrenando modelo...
Epoch 1/20
...
Accuracy en test: 0.XXX
Modelo y componentes guardados exitosamente!
```

### Paso 2: Ejecutar la Aplicación

```bash
# Iniciar servidor Flask
python app.py
```

**Salida esperada:**
```
Modelo cargado exitosamente
🚀 Iniciando aplicación Flask...
* Running on all addresses (0.0.0.0)
* Running on http://127.0.0.1:5000
* Running on http://[IP]:5000
```

### Paso 3: Acceder a la Aplicación

Abre tu navegador web y ve a:
```
http://localhost:5000
```

## 📝 Uso de la Aplicación

1. **Ingresa una reseña** en el área de texto
2. **Haz clic en "Analizar Sentimiento"**
3. **Observa el resultado** con:
   - Clasificación (Positivo/Negativo/Neutral)
   - Porcentaje de confianza
   - Indicadores visuales

### Ejemplos de Reseñas

**Positiva:**
```
Este sistema de gestión es excelente, muy intuitivo y eficiente para nuestro trabajo diario
```

**Neutral:**
```
El software funciona bien, cumple con lo básico pero podría mejorar la interfaz
```

**Negativa:**
```
Esta aplicación es terrible, muy lenta y con muchos errores constantemente
```

## 🔧 Configuración Adicional

### Personalizar el Dataset

Para usar tu propio dataset, reemplaza o modifica `dataset_resenas.csv` con el formato:

```csv
review,label
"Tu reseña aquí",positivo
"Otra reseña",negativo
"Más reseñas",neutral
```

### Modificar el Modelo

Edita `train_model.py` para:
- Cambiar arquitectura de la red neuronal
- Ajustar hiperparámetros
- Modificar preprocesamiento de texto

### Personalizar la Interfaz

Modifica `templates/index.html` y `static/style.css` para cambiar:
- Colores y estilos
- Textos e instrucciones
- Funcionalidades adicionales

## 🛠️ Solución de Problemas

### Error: "No se encontraron los archivos del modelo"

**Solución:** Ejecuta primero el entrenamiento:
```bash
python train_model.py
```

### Error: "ModuleNotFoundError"

**Solución:** Instala las dependencias faltantes:
```bash
pip install [nombre_del_modulo]
```

### Error: "Puerto 5000 en uso"

**Solución:** Cambia el puerto en `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Cambiar puerto
```

### Problemas de Rendimiento

**Solución:** Para datasets grandes:
- Reduce `max_features` en TfidfVectorizer
- Disminuye el número de epochs
- Usa menos neuronas en las capas ocultas

## 📊 Endpoints de la API

### GET `/`
Página principal de la aplicación

### POST `/predecir`
Clasifica una reseña

**Entrada:**
```json
{
  "texto": "Mi reseña sobre el sistema"
}
```

**Salida:**
```json
{
  "sentimiento": "positivo",
  "confianza": 0.85,
  "porcentaje": "85.0%",
  "info": {
    "color": "#28a745",
    "emoji": "😊",
    "descripcion": "Positivo"
  }
}
```

### GET `/salud`
Verifica el estado de la aplicación

**Salida:**
```json
{
  "estado": "activo",
  "modelo_cargado": true
}
```

## 🧪 Desarrollo y Contribución

### Estructura del Código

- **`train_model.py`**: Entrenamiento y guardado del modelo
- **`app.py`**: Servidor Flask y lógica de predicción
- **`templates/index.html`**: Interfaz de usuario
- **`static/style.css`**: Estilos y animaciones

### Agregar Nuevas Características

1. **Nuevas métricas**: Modifica la función `predecir_sentimiento()`
2. **Nuevos endpoints**: Agrega rutas en `app.py`
3. **Mejorar UI**: Edita `index.html` y `style.css`

## 📄 Dependencias Principales

```
Flask==2.3.3
tensorflow==2.13.0
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
joblib==1.3.2
matplotlib==3.7.2
```
