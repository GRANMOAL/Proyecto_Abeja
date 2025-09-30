"""
Módulo de clasificación de imágenes usando Vision Transformer
Para integrar este módulo en el proyecto principal:
1. Instalar: pip install transformers torch pillow
2. Importar en app.py: from classifier import classify_image_with_transformer
3. Reemplazar la función classify_image() en app.py
"""

from transformers import pipeline
from PIL import Image
import base64
from io import BytesIO
import torch

class FlowerClassifier:
    def __init__(self):
        """Inicializa el clasificador de imágenes"""
        print("Cargando modelo Vision Transformer...")
        
        # Verificar si hay GPU disponible
        device = 0 if torch.cuda.is_available() else -1
        
        # Cargar el pipeline de clasificación
        self.classifier = pipeline(
            task="image-classification",
            model="google/vit-base-patch16-224",
            device=device
        )
        
        # Lista de etiquetas relacionadas con flores
        self.flower_labels = [
            'daisy', 'rose', 'sunflower', 'tulip', 'orchid',
            'lily', 'daffodil', 'carnation', 'hibiscus', 'poppy',
            'marigold', 'petunia', 'iris', 'chrysanthemum', 'lavender'
        ]
        
        print(f"Modelo cargado exitosamente (Device: {'GPU' if device == 0 else 'CPU'})")
    
    def is_flower(self, label):
        """Determina si una etiqueta corresponde a una flor"""
        label_lower = label.lower()
        
        # Verificar si contiene palabras clave de flores
        for flower in self.flower_labels:
            if flower in label_lower:
                return True
        
        # Palabras adicionales que indican flores
        flower_keywords = ['flower', 'blossom', 'petal', 'bloom']
        return any(keyword in label_lower for keyword in flower_keywords)
    
    def classify(self, image_data):
        """
        Clasifica una imagen codificada en base64
        
        Args:
            image_data (str): Imagen en formato base64
            
        Returns:
            dict: Resultado de la clasificación con probabilidades
        """
        try:
            # Decodificar imagen base64
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            img_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(img_bytes))
            
            # Convertir a RGB si es necesario
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Clasificar imagen
            results = self.classifier(image)
            
            # Procesar resultados
            top_result = results[0]
            all_results = results[:5]  # Top 5 predicciones
            
            # Determinar si es una flor
            is_flower = self.is_flower(top_result['label'])
            
            # Calcular confianza acumulada para flores
            flower_confidence = sum(
                result['score'] for result in all_results 
                if self.is_flower(result['label'])
            )
            
            return {
                'is_flower': is_flower,
                'label': top_result['label'],
                'score': top_result['score'],
                'flower_confidence': flower_confidence,
                'all_predictions': [
                    {
                        'label': r['label'],
                        'score': r['score'],
                        'is_flower': self.is_flower(r['label'])
                    }
                    for r in all_results
                ]
            }
            
        except Exception as e:
            print(f"Error en clasificación: {e}")
            return {
                'error': str(e),
                'is_flower': False,
                'label': 'error',
                'score': 0.0
            }

# Instancia global del clasificador
_classifier = None

def get_classifier():
    """Obtiene o crea la instancia del clasificador"""
    global _classifier
    if _classifier is None:
        _classifier = FlowerClassifier()
    return _classifier

def classify_image_with_transformer(image_data):
    """
    Función wrapper para usar en app.py
    
    Args:
        image_data (str): Imagen en formato base64
        
    Returns:
        dict: Resultado de la clasificación
    """
    classifier = get_classifier()
    return classifier.classify(image_data)


# Ejemplo de uso
if __name__ == "__main__":
    # Prueba con una imagen
    print("Inicializando clasificador...")
    classifier = FlowerClassifier()
    
    print("\n=== Clasificador de Flores con Vision Transformer ===")
    print("Para usar este módulo:")
    print("1. Asegúrate de tener las dependencias instaladas")
    print("2. Importa en app.py: from classifier import classify_image_with_transformer")
    print("3. Reemplaza la función classify_image() con:")
    print("   result = classify_image_with_transformer(image_data)")
    print("\n=== Ejemplo de Clasificación ===")
    print("Top 5 predicciones con softmax normalizado")
    print("Detección automática de flores por etiquetas")
    print("Confianza acumulada para categorías de flores")