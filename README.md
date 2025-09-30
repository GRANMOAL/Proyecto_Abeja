# 🐝 Reporte Técnico: Sistema de Búsqueda de Caminos con Abejas

## Información del Proyecto

**Nombre del Proyecto:** Bee Pathfinding System  
**Curso:** Inteligencia Artificial  
**Universidad:** Universidad Autónoma del Estado de Hidalgo  
**Carrera:** Ingeniería de Software    
**Fecha:** 09/Octubre/2025   

**Integrantes:**  
  - Fernando Martinez Lopez  
  - Greco Angel Montero Alonso

---

## 1. Resumen Ejecutivo

El presente proyecto implementa un sistema de búsqueda de caminos que simula el comportamiento de una abeja recolectando flores en su trayecto hacia el enjambre. El sistema integra algoritmos de búsqueda DFS (Depth-First Search) y BFS (Breadth-First Search), procesamiento de imágenes con ecualización de histograma, y clasificación de flores mediante técnicas de visión por computadora.

### Características Principales
- Generación dinámica de mundos NxN con obstáculos aleatorios
- Definición interactiva de puntos de inicio y meta
- Implementación de algoritmos DFS y BFS
- Visualización en tiempo real con diseño hexagonal (panal de abeja)
- Clasificación de imágenes de flores
- Sistema de registro y comparación de puntajes
- Interfaz web responsiva con tema nocturno

---

## 2. Objetivos del Proyecto

### 2.1 Objetivo General
Desarrollar un sistema web interactivo que implemente algoritmos de búsqueda informada y no informada, demostrando sus diferencias en escenarios prácticos de recolección de recursos.

### 2.2 Objetivos Específicos
1. Implementar los algoritmos DFS y BFS según las especificaciones académicas
2. Crear una interfaz visual atractiva con diseño hexagonal
3. Integrar técnicas de procesamiento de imágenes
4. Desarrollar un sistema de clasificación de flores
5. Proporcionar análisis comparativo entre algoritmos
6. Registrar y visualizar métricas de rendimiento

---

## 3. Marco Teórico

### 3.1 Algoritmos de Búsqueda

#### 3.1.1 DFS (Búsqueda en Profundidad)
**Definición:** Algoritmo de búsqueda que explora un grafo recorriendo cada rama completamente antes de retroceder.

**Características:**
- Utiliza una estructura de datos tipo pila (stack)
- Complejidad temporal: O(V + E) donde V = vértices, E = aristas
- Complejidad espacial: O(V)
- No garantiza encontrar el camino más corto

**Pseudocódigo:**
```text
función DFS(inicio, meta):
    pila ← [inicio]
    visitados ← conjunto vacío
    
    mientras pila no esté vacía:
        actual ← pila.sacar()
        
        si actual está en visitados:
            continuar
        
        agregar actual a visitados
        
        si actual == meta:
            retornar camino
        
        para cada vecino de actual:
            si vecino no visitado y no es obstáculo:
                pila.agregar(vecino)
    
    retornar null
