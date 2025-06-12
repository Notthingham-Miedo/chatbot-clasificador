# -*- coding: utf-8 -*-
"""
Script para entrenar el modelo de clasificación de sentimientos
y guardar los componentes necesarios para la aplicación Flask
"""

import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical
import joblib
import os

def preprocesar_texto(texto):
    """Preprocesa el texto eliminando caracteres especiales y normalizando"""
    texto = str(texto).lower()  # Convertir a string y minusculizar
    texto = re.sub(r'[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ\s]', '', texto)  # Mantener acentos
    texto = re.sub(r'\s+', ' ', texto).strip()  # Eliminar espacios extra
    return texto

def crear_modelo_demo():
    """Crea un dataset demo si no existe el archivo original"""
    data = {
        'review': [
            'Este sistema es excelente, muy fácil de usar y eficiente',
            'El software funciona bien, cumple con lo esperado',
            'Sistema terrible, muy lento y con muchos errores',
            'Me gusta mucho esta aplicación, muy intuitiva',
            'El programa está bien, nada extraordinario',
            'Horrible experiencia, no lo recomiendo para nada',
            'Funciona correctamente, interfaz amigable',
            'Regular, podría mejorar en algunos aspectos',
            'Excelente herramienta, muy útil para el trabajo',
            'No me gustó, muy complicado de usar'
        ],
        'label': ['positivo', 'neutral', 'negativo', 'positivo', 'neutral', 
                 'negativo', 'positivo', 'neutral', 'positivo', 'negativo']
    }
    return pd.DataFrame(data)

def entrenar_modelo():
    print("Iniciando entrenamiento del modelo...")
    
    # Cargar o crear datos
    try:
        df = pd.read_csv('dataset_resenas.csv')
        print("Dataset cargado desde archivo CSV")
    except FileNotFoundError:
        print("Archivo CSV no encontrado. Creando dataset demo...")
        df = crear_modelo_demo()
        df.to_csv('dataset_resenas.csv', index=False)
        print("Dataset demo creado y guardado")
    
    # Preprocesar texto
    df['review'] = df['review'].apply(preprocesar_texto)
    
    # Vectorización TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['review']).toarray()
    
    # Codificación de etiquetas
    le = LabelEncoder()
    y_encoded = le.fit_transform(df['label'])
    y = to_categorical(y_encoded, num_classes=3)
    
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Crear modelo
    model = Sequential([
        Input(shape=(X.shape[1],)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Entrenar modelo
    print("Entrenando modelo...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluar modelo
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Accuracy en test: {accuracy:.3f}")
    
    # Crear directorio para modelos
    os.makedirs('models', exist_ok=True)
    
    # Guardar modelo y componentes
    model.save('models/modelo_clasificador.h5')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    
    print("Modelo y componentes guardados exitosamente!")
    print("Archivos creados:")
    print("- models/modelo_clasificador.h5")
    print("- models/vectorizer.pkl")
    print("- models/label_encoder.pkl")
    
    return model, vectorizer, le

if __name__ == "__main__":
    entrenar_modelo()