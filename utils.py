"""
Módulo de utilidades para el sistema de búsqueda de caminos
Basado en los scripts proporcionados en clase
"""

import random
from collections import deque
import time

class WorldGenerator:
    """Generador de mundos cuadriculados con obstáculos y flores"""
    
    @staticmethod
    def crear_mundo(n, num_obstaculos=25, num_flores=15, semilla=None):
        """
        Crea un mundo NxN con obstáculos y flores aleatorias
        
        Args:
            n (int): Tamaño de la cuadrícula
            num_obstaculos (int): Número de obstáculos a generar
            num_flores (int): Número de flores a generar
            semilla (int, optional): Semilla para reproducibilidad
            
        Returns:
            tuple: (mundo, obstaculos, flores)
        """
        if semilla is not None:
            random.seed(semilla)
        
        mundo = {}
        for i in range(n):
            for j in range(n):
                mundo[(i, j)] = "empty"
        
        obstaculos = []
        flores = []
        
        # Generar obstáculos en posiciones aleatorias
        intentos = 0
        while len(obstaculos) < num_obstaculos and intentos < num_obstaculos * 3:
            x, y = random.randint(0, n-1), random.randint(0, n-1)
            if mundo[(x, y)] == "empty":
                mundo[(x, y)] = "obstacle"
                obstaculos.append((x, y))
            intentos += 1
        
        # Generar flores en posiciones aleatorias (no en obstáculos)
        intentos = 0
        while len(flores) < num_flores and intentos < num_flores * 3:
            x, y = random.randint(0, n-1), random.randint(0, n-1)
            if mundo[(x, y)] == "empty":
                mundo[(x, y)] = "flower"
                flores.append((x, y))
            intentos += 1
        
        return mundo, obstaculos, flores


class PathFinder:
    """
    Clase para algoritmos de búsqueda de caminos
    Implementa DFS y BFS basados en los scripts de clase
    """
    
    def __init__(self, mundo, n):
        """
        Inicializa el buscador de caminos
        
        Args:
            mundo (dict): Diccionario con el estado del mundo
            n (int): Tamaño de la cuadrícula
        """
        self.mundo = mundo
        self.n = n
        self.estadisticas = {
            'nodos_explorados': 0,
            'tiempo_ejecucion': 0,
            'flores_en_camino': 0
        }
    
    def obtener_vecinos(self, posicion):
        """
        Obtiene los vecinos válidos de una posición
        
        Args:
            posicion (tuple): Posición actual (x, y)
            
        Returns:
            list: Lista de posiciones vecinas válidas
        """
        x, y = posicion
        vecinos = []
        
        # Movimientos: arriba, abajo, izquierda, derecha
        movimientos = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        
        for nx, ny in movimientos:
            if 0 <= nx < self.n and 0 <= ny < self.n:
                if self.mundo.get((nx, ny)) != "obstacle":
                    vecinos.append((nx, ny))
        
        return vecinos
    
    def dfs(self, inicio, meta):
        """
        Búsqueda en profundidad (Depth-First Search)
        Basado en el script 2_agente_basado_modelo_dfs.py
        
        Args:
            inicio (tuple): Posición inicial
            meta (tuple): Posición objetivo
            
        Returns:
            tuple: (camino, flores_recolectadas, estadisticas)
        """
        tiempo_inicio = time.time()
        
        stack = [(inicio, [inicio], set())]
        visitados = set()
        self.estadisticas['nodos_explorados'] = 0
        
        while stack:
            (actual, camino, flores_recolectadas) = stack.pop()
            
            if actual in visitados:
                continue
            
            visitados.add(actual)
            self.estadisticas['nodos_explorados'] += 1
            
            # Verificar si hay flor en la posición actual
            flores_actuales = flores_recolectadas.copy()
            if self.mundo.get(actual) == "flower":
                flores_actuales.add(actual)
            
            # Verificar si llegamos a la meta
            if actual == meta:
                self.estadisticas['tiempo_ejecucion'] = time.time() - tiempo_inicio
                self.estadisticas['flores_en_camino'] = len(flores_actuales)
                return camino, list(flores_actuales), self.estadisticas
            
            # Explorar vecinos
            vecinos = self.obtener_vecinos(actual)
            
            # Agregar vecinos a la pila (en orden inverso para mantener consistencia)
            for vecino in reversed(vecinos):
                if vecino not in visitados:
                    stack.append((vecino, camino + [vecino], flores_actuales))
        
        # No se encontró camino
        self.estadisticas['tiempo_ejecucion'] = time.time() - tiempo_inicio
        return None, [], self.estadisticas
    
    def bfs(self, inicio, meta):
        """
        Búsqueda en amplitud (Breadth-First Search)
        
        Args:
            inicio (tuple): Posición inicial
            meta (tuple): Posición objetivo
            
        Returns:
            tuple: (camino, flores_recolectadas, estadisticas)
        """
        tiempo_inicio = time.time()
        
        queue = deque([(inicio, [inicio], set())])
        visitados = set([inicio])
        self.estadisticas['nodos_explorados'] = 0
        
        while queue:
            (actual, camino, flores_recolectadas) = queue.popleft()
            self.estadisticas['nodos_explorados'] += 1
            
            # Verificar si hay flor en la posición actual
            flores_actuales = flores_recolectadas.copy()
            if self.mundo.get(actual) == "flower":
                flores_actuales.add(actual)
            
            # Verificar si llegamos a la meta
            if actual == meta:
                self.estadisticas['tiempo_ejecucion'] = time.time() - tiempo_inicio
                self.estadisticas['flores_en_camino'] = len(flores_actuales)
                return camino, list(flores_actuales), self.estadisticas
            
            # Explorar vecinos
            vecinos = self.obtener_vecinos(actual)
            
            for vecino in vecinos:
                if vecino not in visitados:
                    visitados.add(vecino)
                    queue.append((vecino, camino + [vecino], flores_actuales))
        
        # No se encontró camino
        self.estadisticas['tiempo_ejecucion'] = time.time() - tiempo_inicio
        return None, [], self.estadisticas
    
    def comparar_algoritmos(self, inicio, meta):
        """
        Compara DFS y BFS en el mismo mundo
        
        Args:
            inicio (tuple): Posición inicial
            meta (tuple): Posición objetivo
            
        Returns:
            dict: Comparación de resultados
        """
        # Ejecutar DFS
        camino_dfs, flores_dfs, stats_dfs = self.dfs(inicio, meta)
        
        # Ejecutar BFS
        camino_bfs, flores_bfs, stats_bfs = self.bfs(inicio, meta)
        
        comparacion = {
            'dfs': {
                'camino_encontrado': camino_dfs is not None,
                'longitud_camino': len(camino_dfs) if camino_dfs else 0,
                'flores_recolectadas': len(flores_dfs),
                'nodos_explorados': stats_dfs['nodos_explorados'],
                'tiempo': stats_dfs['tiempo_ejecucion'],
                'eficiencia': len(flores_dfs) / len(camino_dfs) if camino_dfs else 0
            },
            'bfs': {
                'camino_encontrado': camino_bfs is not None,
                'longitud_camino': len(camino_bfs) if camino_bfs else 0,
                'flores_recolectadas': len(flores_bfs),
                'nodos_explorados': stats_bfs['nodos_explorados'],
                'tiempo': stats_bfs['tiempo_ejecucion'],
                'eficiencia': len(flores_bfs) / len(camino_bfs) if camino_bfs else 0
            }
        }
        
        # Determinar ganador
        if comparacion['dfs']['flores_recolectadas'] > comparacion['bfs']['flores_recolectadas']:
            comparacion['ganador'] = 'DFS'
            comparacion['razon'] = 'Recolectó más flores'
        elif comparacion['bfs']['flores_recolectadas'] > comparacion['dfs']['flores_recolectadas']:
            comparacion['ganador'] = 'BFS'
            comparacion['razon'] = 'Recolectó más flores'
        elif comparacion['bfs']['longitud_camino'] < comparacion['dfs']['longitud_camino']:
            comparacion['ganador'] = 'BFS'
            comparacion['razon'] = 'Camino más corto con mismas flores'
        else:
            comparacion['ganador'] = 'Empate'
            comparacion['razon'] = 'Resultados similares'
        
        return comparacion


class ImageProcessor:
    """
    Procesamiento de imágenes con ecualización de histograma
    """
    
    @staticmethod
    def ecualizar_histograma(imagen_array):
        """
        Aplica ecualización de histograma para mejorar imágenes subexpuestas
        
        Args:
            imagen_array (numpy.array): Imagen en formato numpy array
            
        Returns:
            numpy.array: Imagen ecualizada
        """
        import cv2
        import numpy as np
        
        if len(imagen_array.shape) == 3:
            # Imagen a color - convertir a YCrCb y ecualizar solo Y
            img_ycrcb = cv2.cvtColor(imagen_array, cv2.COLOR_RGB2YCrCb)
            img_ycrcb[:, :, 0] = cv2.equalizeHist(img_ycrcb[:, :, 0])
            img_ecualizada = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2RGB)
        else:
            # Imagen en escala de grises
            img_ecualizada = cv2.equalizeHist(imagen_array)
        
        return img_ecualizada
    
    @staticmethod
    def aplicar_clahe(imagen_array, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Para mejor resultado en imágenes con iluminación no uniforme
        
        Args:
            imagen_array (numpy.array): Imagen en formato numpy array
            clip_limit (float): Límite de contraste
            tile_grid_size (tuple): Tamaño de la cuadrícula
            
        Returns:
            numpy.array: Imagen con CLAHE aplicado
        """
        import cv2
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        if len(imagen_array.shape) == 3:
            img_ycrcb = cv2.cvtColor(imagen_array, cv2.COLOR_RGB2YCrCb)
            img_ycrcb[:, :, 0] = clahe.apply(img_ycrcb[:, :, 0])
            img_mejorada = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2RGB)
        else:
            img_mejorada = clahe.apply(imagen_array)
        
        return img_mejorada


class ScoreManager:
    """
    Gestor de puntajes y estadísticas
    """
    
    def __init__(self):
        self.scores = {
            'dfs': [],
            'bfs': []
        }
        self.historial_completo = []
    
    def agregar_score(self, algoritmo, datos):
        """
        Agrega un nuevo puntaje
        
        Args:
            algoritmo (str): 'dfs' o 'bfs'
            datos (dict): Datos del recorrido
        """
        score = {
            'algoritmo': algoritmo.upper(),
            'longitud_camino': datos.get('longitud_camino', 0),
            'flores_recolectadas': datos.get('flores_recolectadas', 0),
            'tiempo': datos.get('tiempo', 0),
            'nodos_explorados': datos.get('nodos_explorados', 0),
            'timestamp': time.time()
        }
        
        self.scores[algoritmo].append(score)
        self.historial_completo.append(score)
    
    def obtener_promedios(self, algoritmo):
        """
        Calcula promedios para un algoritmo
        
        Args:
            algoritmo (str): 'dfs' o 'bfs'
            
        Returns:
            dict: Promedios calculados
        """
        scores = self.scores[algoritmo]
        
        if not scores:
            return {
                'longitud_camino': 0,
                'flores_recolectadas': 0,
                'tiempo': 0,
                'eficiencia': 0
            }
        
        total_camino = sum(s['longitud_camino'] for s in scores)
        total_flores = sum(s['flores_recolectadas'] for s in scores)
        total_tiempo = sum(s['tiempo'] for s in scores)
        n = len(scores)
        
        avg_camino = total_camino / n
        avg_flores = total_flores / n
        
        return {
            'longitud_camino': round(avg_camino, 2),
            'flores_recolectadas': round(avg_flores, 2),
            'tiempo': round(total_tiempo / n, 4),
            'eficiencia': round(avg_flores / avg_camino if avg_camino > 0 else 0, 3),
            'total_ejecuciones': n
        }
    
    def generar_reporte_comparativo(self):
        """
        Genera un reporte comparativo completo
        
        Returns:
            dict: Reporte con comparación detallada
        """
        dfs_stats = self.obtener_promedios('dfs')
        bfs_stats = self.obtener_promedios('bfs')
        
        reporte = {
            'dfs': dfs_stats,
            'bfs': bfs_stats,
            'total_ejecuciones': len(self.historial_completo),
            'analisis': {}
        }
        
        # Análisis comparativo
        if dfs_stats['flores_recolectadas'] > bfs_stats['flores_recolectadas']:
            reporte['analisis']['mejor_recoleccion'] = 'DFS'
        elif bfs_stats['flores_recolectadas'] > dfs_stats['flores_recolectadas']:
            reporte['analisis']['mejor_recoleccion'] = 'BFS'
        else:
            reporte['analisis']['mejor_recoleccion'] = 'Empate'
        
        if dfs_stats['longitud_camino'] < bfs_stats['longitud_camino']:
            reporte['analisis']['camino_mas_corto'] = 'DFS'
        elif bfs_stats['longitud_camino'] < dfs_stats['longitud_camino']:
            reporte['analisis']['camino_mas_corto'] = 'BFS'
        else:
            reporte['analisis']['camino_mas_corto'] = 'Empate'
        
        if dfs_stats['eficiencia'] > bfs_stats['eficiencia']:
            reporte['analisis']['mas_eficiente'] = 'DFS'
        elif bfs_stats['eficiencia'] > dfs_stats['eficiencia']:
            reporte['analisis']['mas_eficiente'] = 'BFS'
        else:
            reporte['analisis']['mas_eficiente'] = 'Empate'
        
        # Recomendación
        if reporte['analisis']['mejor_recoleccion'] == 'DFS':
            reporte['recomendacion'] = 'DFS es mejor para maximizar la recolección de flores'
        elif reporte['analisis']['camino_mas_corto'] == 'BFS':
            reporte['recomendacion'] = 'BFS es mejor para encontrar el camino más eficiente'
        else:
            reporte['recomendacion'] = 'Ambos algoritmos tienen rendimiento similar'
        
        return reporte
    
    def limpiar_scores(self):
        """Limpia todos los puntajes registrados"""
        self.scores = {'dfs': [], 'bfs': []}
        self.historial_completo = []


# Funciones de utilidad adicionales
def validar_posicion(posicion, n, mundo, tipo='general'):
    """
    Valida que una posición sea válida para un propósito específico
    
    Args:
        posicion (tuple): Posición a validar
        n (int): Tamaño de la cuadrícula
        mundo (dict): Estado del mundo
        tipo (str): Tipo de validación ('general', 'inicio', 'meta')
        
    Returns:
        tuple: (es_valida, mensaje)
    """
    x, y = posicion
    
    # Verificar límites
    if not (0 <= x < n and 0 <= y < n):
        return False, "Posición fuera de los límites"
    
    # Verificar si es obstáculo
    if mundo.get(posicion) == "obstacle":
        return False, "No se puede colocar en un obstáculo"
    
    # Validaciones específicas por tipo
    if tipo == 'inicio' or tipo == 'meta':
        # Las posiciones de inicio y meta pueden estar en flores
        return True, "Posición válida"
    
    return True, "Posición válida"


def calcular_distancia_manhattan(pos1, pos2):
    """
    Calcula la distancia Manhattan entre dos posiciones
    
    Args:
        pos1 (tuple): Primera posición
        pos2 (tuple): Segunda posición
        
    Returns:
        int: Distancia Manhattan
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def hay_camino_posible(mundo, inicio, meta, n):
    """
    Verifica rápidamente si existe un camino posible entre inicio y meta
    Usando BFS simplificado
    
    Args:
        mundo (dict): Estado del mundo
        inicio (tuple): Posición inicial
        meta (tuple): Posición objetivo
        n (int): Tamaño de la cuadrícula
        
    Returns:
        bool: True si existe camino posible
    """
    if inicio == meta:
        return True
    
    queue = deque([inicio])
    visitados = set([inicio])
    
    while queue:
        actual = queue.popleft()
        
        x, y = actual
        vecinos = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        
        for nx, ny in vecinos:
            if 0 <= nx < n and 0 <= ny < n:
                if (nx, ny) == meta:
                    return True
                
                if (nx, ny) not in visitados and mundo.get((nx, ny)) != "obstacle":
                    visitados.add((nx, ny))
                    queue.append((nx, ny))
    
    return False