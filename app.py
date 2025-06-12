from flask import Flask, render_template, request, jsonify
import numpy as np
import re
import joblib
from keras.models import load_model
import os

app = Flask(__name__)

# Variables globales para el modelo
model = None
vectorizer = None
label_encoder = None

def preprocesar_texto(texto):
    """Preprocesa el texto de la misma forma que en el entrenamiento"""
    texto = str(texto).lower()
    texto = re.sub(r'[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def cargar_modelo():
    """Carga el modelo y componentes necesarios"""
    global model, vectorizer, label_encoder
    
    try:
        model = load_model('models/modelo_clasificador.h5')
        vectorizer = joblib.load('models/vectorizer.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        print("Modelo cargado exitosamente")
        return True
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return False

def predecir_sentimiento(texto):
    """Predice el sentimiento de un texto"""
    if not all([model, vectorizer, label_encoder]):
        return None, None
    
    # Preprocesar texto
    texto_procesado = preprocesar_texto(texto)
    
    # Vectorizar
    texto_vectorizado = vectorizer.transform([texto_procesado]).toarray()
    
    # Predecir
    prediccion = model.predict(texto_vectorizado)
    clase_predicha = np.argmax(prediccion[0])
    confianza = np.max(prediccion[0])
    
    # Decodificar etiqueta
    etiqueta = label_encoder.inverse_transform([clase_predicha])[0]
    
    return etiqueta, confianza

@app.route('/')
def home():
    """Página principal"""
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    """Endpoint para realizar predicciones"""
    try:
        data = request.get_json()
        texto = data.get('texto', '').strip()
        
        if not texto:
            return jsonify({
                'error': 'Por favor ingresa una reseña válida'
            }), 400
        
        etiqueta, confianza = predecir_sentimiento(texto)
        
        if etiqueta is None:
            return jsonify({
                'error': 'Error en el modelo de predicción'
            }), 500
        
        # Mapear etiquetas a colores y emojis
        info_sentimiento = {
            'positivo': {
                'color': '#28a745',
                'emoji': '😊',
                'descripcion': 'Positivo'
            },
            'negativo': {
                'color': '#dc3545',
                'emoji': '😞',
                'descripcion': 'Negativo'
            },
            'neutral': {
                'color': '#ffc107',
                'emoji': '😐',
                'descripcion': 'Neutral'
            }
        }
        
        resultado = {
            'sentimiento': etiqueta,
            'confianza': float(confianza),
            'porcentaje': f"{confianza * 100:.1f}%",
            'info': info_sentimiento.get(etiqueta, {})
        }
        
        return jsonify(resultado)
        
    except Exception as e:
        return jsonify({
            'error': f'Error al procesar la solicitud: {str(e)}'
        }), 500

@app.route('/salud')
def salud():
    """Endpoint para verificar el estado de la aplicación"""
    modelo_cargado = all([model, vectorizer, label_encoder])
    return jsonify({
        'estado': 'activo',
        'modelo_cargado': modelo_cargado
    })

if __name__ == '__main__':
    # Verificar si existen los archivos del modelo
    archivos_modelo = [
        'models/modelo_clasificador.h5',
        'models/vectorizer.pkl',
        'models/label_encoder.pkl'
    ]
    
    if not all(os.path.exists(archivo) for archivo in archivos_modelo):
        print("⚠️  No se encontraron los archivos del modelo.")
        print("Ejecuta primero: python train_model.py")
        exit(1)
    
    # Cargar modelo
    if cargar_modelo():
        print("🚀 Iniciando aplicación Flask...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ No se pudo cargar el modelo. Verifica los archivos.")
        exit(1)