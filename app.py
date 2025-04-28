import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging

# Flask uygulaması için konfigürasyon
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_PATH'] = 'model/grape_disease_model.h5'

# Logging yapılandırması
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Klasörlerin varlığını kontrol et
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('model', exist_ok=True)

# Hastalık sınıfları
CLASSES = ['black_rot', 'esca', 'healthy', 'leaf_blight', 'powdery_mildew']

# Görüntü boyutu
IMG_SIZE = 224

def train_model():
    """
    Üzüm yaprağı hastalık sınıflandırma modelini eğitir
    """
    logger.info("Model eğitimi başlatılıyor...")
    
    # Veri artırma ile eğitim veri üreteci
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Eğitim ve doğrulama verilerini yükle
    train_generator = train_datagen.flow_from_directory(
        'dataset',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = train_datagen.flow_from_directory(
        'dataset',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Önceden eğitilmiş MobileNetV2 modelini temel olarak kullan
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False  # Başlangıçta temel modeli dondur
    
    # Sınıflandırma kafası oluştur
    model = Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
    
    # Modeli derle
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Early stopping ekle
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Modeli eğit
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        callbacks=[early_stopping]
    )
    
    # Son modeli kaydet
    model.save(app.config['MODEL_PATH'])
    logger.info(f"Model başarıyla kaydedildi: {app.config['MODEL_PATH']}")
    
    # Eğitim performansını görselleştir
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Doğruluk')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend(['Eğitim', 'Doğrulama'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend(['Eğitim', 'Doğrulama'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('model/training_history.png')
    
    return model

def preprocess_image(img_path):
    """
    Görüntüyü model için hazırlar
    """
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_disease(img_path, model=None):
    """
    Verilen görüntüde üzüm yaprağı hastalığını tahmin eder
    """
    # Model yüklenmemişse yükle
    if model is None:
        try:
            model = load_model(app.config['MODEL_PATH'])
            logger.info("Model başarıyla yüklendi.")
        except Exception as e:
            logger.error(f"Model yüklenirken hata oluştu: {str(e)}")
            return None, str(e)
    
    # Görüntüyü işle
    try:
        processed_img = preprocess_image(img_path)
        predictions = model.predict(processed_img)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = CLASSES[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        result = {
            'disease': predicted_class,
            'confidence': confidence,
            'all_probabilities': {cls: float(prob) for cls, prob in zip(CLASSES, predictions[0])}
        }
        
        return result, None
    except Exception as e:
        logger.error(f"Tahmin sırasında hata oluştu: {str(e)}")
        return None, str(e)

# API Endpoint'leri
@app.route('/predict', methods=['POST'])
def predict():
    """
    Yüklenen görüntü için hastalık tahmini yapan API endpoint'i
    """
    if 'image' not in request.files:
        return jsonify({'error': 'Görüntü dosyası bulunamadı'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        result, error = predict_disease(file_path)
        
        if error:
            return jsonify({'error': error}), 500
            
        return jsonify(result), 200

@app.route('/train', methods=['POST'])
def start_training():
    """
    Model eğitimini başlatan API endpoint'i
    """
    try:
        train_model()
        return jsonify({'message': 'Model eğitimi başarıyla tamamlandı!'}), 200
    except Exception as e:
        logger.error(f"Eğitim sırasında hata oluştu: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    API'nin çalışıp çalışmadığını kontrol eden endpoint
    """
    return jsonify({'status': 'API çalışıyor!'}), 200

if __name__ == '__main__':
    if not os.path.exists(app.config['MODEL_PATH']):
        logger.info("Mevcut model bulunamadı. Eğitim başlatılıyor...")
        train_model()
    app.run(debug=True, host='0.0.0.0', port=5000) 
