from flask import Flask, render_template, request, jsonify
import random
import base64
from io import BytesIO
from PIL import Image, ImageEnhance
import numpy as np
from collections import deque
import cv2

app = Flask(__name__)

# Configuración del mundo
N = 12  # Tamaño de la cuadrícula
world_state = {
    'grid': {},
    'start': None,
    'goal': None,
    'flowers': [],
    'obstacles': [],
    'scores': {'dfs': [], 'bfs': []}
}

def create_world(n, num_obstacles=25, num_flowers=15):
    """Crea un mundo NxN con obstáculos y flores aleatorias"""
    world = {}
    for i in range(n):
        for j in range(n):
            world[(i, j)] = "empty"
    
    obstacles = []
    flowers = []
    
    # Generar obstáculos aleatorios
    for _ in range(num_obstacles):
        x, y = random.randint(0, n-1), random.randint(0, n-1)
        world[(x, y)] = "obstacle"
        obstacles.append((x, y))
    
    # Generar flores aleatorias
    for _ in range(num_flowers):
        x, y = random.randint(0, n-1), random.randint(0, n-1)
        if world[(x, y)] == "empty":
            world[(x, y)] = "flower"
            flowers.append((x, y))
    
    return world, obstacles, flowers

def dfs_search(world, start, goal, n):
    """Búsqueda en profundidad"""
    stack = [(start, [start], [])]
    visited = set()
    
    while stack:
        (current, path, flowers_collected) = stack.pop()
        
        if current in visited:
            continue
        
        visited.add(current)
        
        # Verificar si hay flor en la posición actual
        current_flowers = flowers_collected.copy()
        if world.get(current) == "flower" and current not in flowers_collected:
            current_flowers.append(current)
        
        if current == goal:
            return path, current_flowers
        
        x, y = current
        # Movimientos: arriba, abajo, izquierda, derecha
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        
        for nx, ny in neighbors:
            if 0 <= nx < n and 0 <= ny < n:
                if (nx, ny) not in visited and world.get((nx, ny)) != "obstacle":
                    stack.append(((nx, ny), path + [(nx, ny)], current_flowers))
    
    return None, []

def bfs_search(world, start, goal, n):
    """Búsqueda en amplitud"""
    queue = deque([(start, [start], [])])
    visited = set([start])
    
    while queue:
        (current, path, flowers_collected) = queue.popleft()
        
        # Verificar si hay flor en la posición actual
        current_flowers = flowers_collected.copy()
        if world.get(current) == "flower" and current not in flowers_collected:
            current_flowers.append(current)
        
        if current == goal:
            return path, current_flowers
        
        x, y = current
        # Movimientos: arriba, abajo, izquierda, derecha
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        
        for nx, ny in neighbors:
            if 0 <= nx < n and 0 <= ny < n:
                if (nx, ny) not in visited and world.get((nx, ny)) != "obstacle":
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)], current_flowers))
    
    return None, []

def equalize_histogram(image_data):
    """Ecualización de histograma para mejorar imágenes subexpuestas"""
    try:
        # Decodificar imagen base64
        img_bytes = base64.b64decode(image_data.split(',')[1])
        img = Image.open(BytesIO(img_bytes))
        
        # Convertir a numpy array
        img_array = np.array(img)
        
        # Si es RGB, convertir a YCrCb para ecualizar solo luminancia
        if len(img_array.shape) == 3:
            img_ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
            img_ycrcb[:, :, 0] = cv2.equalizeHist(img_ycrcb[:, :, 0])
            img_equalized = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2RGB)
        else:
            img_equalized = cv2.equalizeHist(img_array)
        
        # Convertir de vuelta a base64
        img_pil = Image.fromarray(img_equalized)
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        return None

def classify_image(image_data):
    """Clasificación simple de imágenes (simulación)"""
    # En producción, aquí usarías un transformador como en tu código de ejemplo
    # Por ahora, simulamos la clasificación
    
    classifications = [
        {"label": "flower", "score": 0.85},
        {"label": "rose", "score": 0.78},
        {"label": "sunflower", "score": 0.72},
        {"label": "tulip", "score": 0.68},
        {"label": "daisy", "score": 0.65}
    ]
    
    # Seleccionar aleatoriamente si es flor o no
    is_flower = random.choice([True, False])
    
    if is_flower:
        return random.choice(classifications[:4])
    else:
        return {"label": "not_a_flower", "score": 0.92}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/initialize_world', methods=['POST'])
def initialize_world():
    """Inicializa un nuevo mundo"""
    global world_state
    
    data = request.json
    n = data.get('size', N)
    
    world, obstacles, flowers = create_world(n)
    
    world_state['grid'] = world
    world_state['obstacles'] = obstacles
    world_state['flowers'] = flowers
    world_state['start'] = None
    world_state['goal'] = None
    
    return jsonify({
        'success': True,
        'size': n,
        'obstacles': obstacles,
        'flowers': flowers
    })

@app.route('/set_position', methods=['POST'])
def set_position():
    """Define punto de inicio o meta"""
    global world_state
    
    data = request.json
    pos_type = data.get('type')  # 'start' o 'goal'
    position = tuple(data.get('position'))
    
    if world_state['grid'].get(position) == "obstacle":
        return jsonify({'success': False, 'message': 'No se puede colocar en un obstáculo'})
    
    world_state[pos_type] = position
    
    return jsonify({'success': True, 'position': position})

@app.route('/find_path', methods=['POST'])
def find_path():
    """Encuentra un camino usando DFS o BFS"""
    global world_state
    
    data = request.json
    algorithm = data.get('algorithm')  # 'dfs' o 'bfs'
    
    if not world_state['start'] or not world_state['goal']:
        return jsonify({'success': False, 'message': 'Define inicio y meta primero'})
    
    if algorithm == 'dfs':
        path, flowers = dfs_search(
            world_state['grid'],
            world_state['start'],
            world_state['goal'],
            N
        )
    else:
        path, flowers = bfs_search(
            world_state['grid'],
            world_state['start'],
            world_state['goal'],
            N
        )
    
    if path:
        score = {
            'algorithm': algorithm.upper(),
            'path_length': len(path),
            'flowers_collected': len(flowers),
            'flowers': flowers
        }
        world_state['scores'][algorithm].append(score)
        
        return jsonify({
            'success': True,
            'path': path,
            'flowers_collected': flowers,
            'score': score
        })
    else:
        return jsonify({'success': False, 'message': 'No se encontró camino'})

@app.route('/get_scores', methods=['GET'])
def get_scores():
    """Obtiene los puntajes registrados"""
    return jsonify(world_state['scores'])

@app.route('/equalize_image', methods=['POST'])
def equalize_image():
    """Ecualiza el histograma de una imagen"""
    data = request.json
    image_data = data.get('image')
    
    equalized = equalize_histogram(image_data)
    
    if equalized:
        return jsonify({'success': True, 'equalized_image': equalized})
    else:
        return jsonify({'success': False, 'message': 'Error al ecualizar imagen'})

@app.route('/classify_flower', methods=['POST'])
def classify_flower():
    """Clasifica una imagen de flor"""
    data = request.json
    image_data = data.get('image')
    
    result = classify_image(image_data)
    
    return jsonify({
        'success': True,
        'classification': result
    })

if __name__ == '__main__':
    app.run(debug=True)