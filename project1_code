"""
PROYECTO 1 - INTELIGENCIA ARTIFICIAL
Inteligencia Artificial - Grupo 80

Tema: Conecta 4 Inteligente (Modo Puzzle + Modo Duelo)
Algoritmos: DFS, BFS, UCS, A* y Minimax con Poda Alfa-Beta
Integrantes:
 - Santiago David Guerrero Jaramillo - 2419030
 - Hugart Andres Estrada Velasco - 2610126
"""



import time
import heapq

FILAS = 6
COLUMNAS = 7
# Costos distintos por columna (UCS y A*)
COSTO_COLUMNA = [1.0, 1.5, 2.0, 3.0, 2.0, 1.5, 1.0]
C_MIN = 1.0

def crear_tablero():
    return tuple(tuple(' ' for _ in range(COLUMNAS)) for _ in range(FILAS))

def imprimir_tablero(t):
    print("\n  0   1   2   3   4   5   6")
    print("+---+---+---+---+---+---+---+")
    for r in range(FILAS):
        fila_str = "| " + " | ".join(t[r]) + " |"
        print(fila_str)
        print("+---+---+---+---+---+---+---+")
    print("Costos: [1.0, 1.5, 2.0, 3.0, 2.0, 1.5, 1.0]\n")

def soltar_ficha(tablero, col, ficha):
    """Devuelve un nuevo tablero con la ficha colocada o None si la columna está llena."""
    if tablero[0][col] != ' ':
        return None
    nuevo = [list(fila) for fila in tablero]
    for r in reversed(range(FILAS)):
        if nuevo[r][col] == ' ':
            nuevo[r][col] = ficha
            break
    return tuple(tuple(fila) for fila in nuevo)

def es_meta(t, ficha):
    """Verifica si hay 4 en línea de la ficha indicada."""
    # Horizontal
    for r in range(FILAS):
        for c in range(COLUMNAS - 3):
            if all(t[r][c+i] == ficha for i in range(4)): return True
    # Vertical
    for r in range(FILAS - 3):
        for c in range(COLUMNAS):
            if all(t[r+i][c] == ficha for i in range(4)): return True
    # Diagonal Positiva (\)
    for r in range(FILAS - 3):
        for c in range(COLUMNAS - 3):
            if all(t[r+i][c+i] == ficha for i in range(4)): return True
    # Diagonal Negativa (/)
    for r in range(3, FILAS):
        for c in range(COLUMNAS - 3):
            if all(t[r-i][c+i] == ficha for i in range(4)): return True
    return False

# ------------------------------------------------------------------------------
# HEURÍSTICA PROPIA PARA A* (ADMISIBLE Y CONSISTENTE)
# ------------------------------------------------------------------------------

def heuristica_conecta4(t, ficha):
    """
    Calcula cuántas fichas faltan para completar el 4 en línea más cercano
    y lo multiplica por el costo mínimo de tiro (c_min).
    """
    if es_meta(t, ficha):
        return 0.0

    max_alineadas = 0
    # Revisar todas las ventanas posibles de 4
    for r in range(FILAS):
        for c in range(COLUMNAS):
            direcciones = [(0,1), (1,0), (1,1), (-1,1)]
            for dr, dc in direcciones:
                if 0 <= r + 3*dr < FILAS and 0 <= c + 3*dc < COLUMNAS:
                    ventana = [t[r + i*dr][c + i*dc] for i in range(4)]
                    # Ventana válida si solo tiene fichas mías o vacías
                    if ventana.count(' ') + ventana.count(ficha) == 4:
                        max_alineadas = max(max_alineadas, ventana.count(ficha))

    fichas_faltantes = 4 - max_alineadas
    return fichas_faltantes * C_MIN

# ------------------------------------------------------------------------------
# ALGORITMOS DE BÚSQUEDA EN UN SOLO AGENTE (PUZZLE)
# ------------------------------------------------------------------------------

def bfs(inicio, ficha='O'):
    t_ini = time.perf_counter()
    cola = [(inicio, [], 0.0)]
    visitados = {inicio}
    nodos_exp = 0

    while cola:
        tab, cols_usadas, costo = cola.pop(0)
        nodos_exp += 1

        if es_meta(tab, ficha):
            return cols_usadas, costo, nodos_exp, time.perf_counter() - t_ini

        for c in range(COLUMNAS):
            nuevo_tab = soltar_ficha(tab, c, ficha)
            if nuevo_tab and nuevo_tab not in visitados:
                visitados.add(nuevo_tab)
                cola.append((nuevo_tab, cols_usadas + [c], costo + COSTO_COLUMNA[c]))
    return None, 0, nodos_exp, 0

def dfs(inicio, ficha='O', limite_prof=5):
    t_ini = time.perf_counter()
    pila = [(inicio, [], 0.0)]
    visitados = set()
    nodos_exp = 0

    while pila:
        tab, cols_usadas, costo = pila.pop()
        if tab in visitados or len(cols_usadas) > limite_prof:
            continue
        visitados.add(tab)
        nodos_exp += 1

        if es_meta(tab, ficha):
            return cols_usadas, costo, nodos_exp, time.perf_counter() - t_ini

        for c in range(COLUMNAS):
            nuevo_tab = soltar_ficha(tab, c, ficha)
            if nuevo_tab and nuevo_tab not in visitados:
                pila.append((nuevo_tab, cols_usadas + [c], costo + COSTO_COLUMNA[c]))
    return None, 0, nodos_exp, 0

def ucs(inicio, ficha='O'):
    t_ini = time.perf_counter()
    # (costo_acumulado, contador_desempate, tablero, camino)
    frontera = [(0.0, 0, inicio, [])]
    costos = {inicio: 0.0}
    nodos_exp = 0
    cnt = 0

    while frontera:
        g, _, tab, cols_usadas = heapq.heappop(frontera)
        nodos_exp += 1

        if es_meta(tab, ficha):
            return cols_usadas, g, nodos_exp, time.perf_counter() - t_ini

        if g > costos.get(tab, float('inf')):
            continue

        for c in range(COLUMNAS):
            nuevo_tab = soltar_ficha(tab, c, ficha)
            if nuevo_tab:
                nuevo_g = g + COSTO_COLUMNA[c]
                if nuevo_g < costos.get(nuevo_tab, float('inf')):
                    costos[nuevo_tab] = nuevo_g
                    cnt += 1
                    heapq.heappush(frontera, (nuevo_g, cnt, nuevo_tab, cols_usadas + [c]))
    return None, 0, nodos_exp, 0

def a_estrella(inicio, ficha='O'):
    t_ini = time.perf_counter()
    h_ini = heuristica_conecta4(inicio, ficha)
    frontera = [(h_ini, 0.0, 0, inicio, [])]
    costos_g = {inicio: 0.0}
    nodos_exp = 0
    cnt = 0

    while frontera:
        f, g, _, tab, cols_usadas = heapq.heappop(frontera)
        nodos_exp += 1

        if es_meta(tab, ficha):
            return cols_usadas, g, nodos_exp, time.perf_counter() - t_ini

        if g > costos_g.get(tab, float('inf')):
            continue

        for c in range(COLUMNAS):
            nuevo_tab = soltar_ficha(tab, c, ficha)
            if nuevo_tab:
                nuevo_g = g + COSTO_COLUMNA[c]
                if nuevo_g < costos_g.get(nuevo_tab, float('inf')):
                    costos_g[nuevo_tab] = nuevo_g
                    nuevo_f = nuevo_g + heuristica_conecta4(nuevo_tab, ficha)
                    cnt += 1
                    heapq.heappush(frontera, (nuevo_f, nuevo_g, cnt, nuevo_tab, cols_usadas + [c]))
    return None, 0, nodos_exp, 0

# ------------------------------------------------------------------------------
# BÚSQUEDA ADVERSARIA: MINIMAX CON PODA ALFA-BETA
# ------------------------------------------------------------------------------

nodos_evaluados = 0

def evaluar_ventana(ventana, ficha_ia='O', ficha_rival='X'):
    puntaje = 0
    if ventana.count(ficha_ia) == 4: puntaje += 1000
    elif ventana.count(ficha_ia) == 3 and ventana.count(' ') == 1: puntaje += 50
    elif ventana.count(ficha_ia) == 2 and ventana.count(' ') == 2: puntaje += 10

    if ventana.count(ficha_rival) == 3 and ventana.count(' ') == 1: puntaje -= 80
    return puntaje

def funcion_evaluacion(t):
    puntaje = 0
    # Centro más valioso
    col_centro = [t[r][3] for r in range(FILAS)]
    puntaje += col_centro.count('O') * 6

    # Evaluar todas las ventanas de 4
    for r in range(FILAS):
        for c in range(COLUMNAS):
            for dr, dc in [(0,1), (1,0), (1,1), (-1,1)]:
                if 0 <= r + 3*dr < FILAS and 0 <= c + 3*dc < COLUMNAS:
                    ventana = [t[r + i*dr][c + i*dc] for i in range(4)]
                    puntaje += evaluar_ventana(ventana)
    return puntaje

def minimax_sin_poda(tab, prof, es_max):
    global nodos_evaluados
    nodos_evaluados += 1

    if es_meta(tab, 'O'): return 10000
    if es_meta(tab, 'X'): return -10000
    if prof == 0 or all(tab[0][c] != ' ' for c in range(COLUMNAS)):
        return funcion_evaluacion(tab)

    if es_max:
        mejor = -float('inf')
        for c in range(COLUMNAS):
            sig = soltar_ficha(tab, c, 'O')
            if sig:
                mejor = max(mejor, minimax_sin_poda(sig, prof - 1, False))
        return mejor
    else:
        mejor = float('inf')
        for c in range(COLUMNAS):
            sig = soltar_ficha(tab, c, 'X')
            if sig:
                mejor = min(mejor, minimax_sin_poda(sig, prof - 1, True))
        return mejor

def minimax_con_poda(tab, prof, alfa, beta, es_max):
    global nodos_evaluados
    nodos_evaluados += 1

    if es_meta(tab, 'O'): return 10000
    if es_meta(tab, 'X'): return -10000
    if prof == 0 or all(tab[0][c] != ' ' for c in range(COLUMNAS)):
        return funcion_evaluacion(tab)

    if es_max:
        mejor = -float('inf')
        for c in range(COLUMNAS):
            sig = soltar_ficha(tab, c, 'O')
            if sig:
                mejor = max(mejor, minimax_con_poda(sig, prof - 1, alfa, beta, False))
                alfa = max(alfa, mejor)
                if beta <= alfa:
                    break  # Poda beta
        return mejor
    else:
        mejor = float('inf')
        for c in range(COLUMNAS):
            sig = soltar_ficha(tab, c, 'X')
            if sig:
                mejor = min(mejor, minimax_con_poda(sig, prof - 1, alfa, beta, True))
                beta = min(beta, mejor)
                if beta <= alfa:
                    break  # Poda alfa
        return mejor

def mejor_columna_ia(tab, prof=4):
    global nodos_evaluados
    # 1. Con Poda
    nodos_evaluados = 0
    mejor_col = 3
    mejor_val = -float('inf')
    alfa, beta = -float('inf'), float('inf')
    for c in range(COLUMNAS):
        sig = soltar_ficha(tab, c, 'O')
        if sig:
            val = minimax_con_poda(sig, prof - 1, alfa, beta, False)
            if val > mejor_val:
                mejor_val = val
                mejor_col = c
            alfa = max(alfa, mejor_val)
    nodos_con = nodos_evaluados

    # 2. Sin Poda (Para contrastar en la rúbrica)
    nodos_evaluados = 0
    for c in range(COLUMNAS):
        sig = soltar_ficha(tab, c, 'O')
        if sig:
            _ = minimax_sin_poda(sig, prof - 1, False)
    nodos_sin = nodos_evaluados

    return mejor_col, nodos_con, nodos_sin

# ------------------------------------------------------------------------------
# MENÚS E INTERFAZ
# ------------------------------------------------------------------------------

def crear_tablero_puzzle():
    """Genera un estado inicial con fichas ya colocadas para el desafío de búsqueda."""
    t = crear_tablero()
    # Colocar algunas fichas base para obligar a buscar
    for c in [0, 1, 2, 4]: t = soltar_ficha(t, c, 'X')
    for c in [0, 1]: t = soltar_ficha(t, c, 'O')
    return t

def menu_puzzle():
    tab = crear_tablero_puzzle()
    print("\n--- MODO PUZZLE (ALGORITMOS DE BÚSQUEDA) ---")
    print("Tablero inicial con reto:")
    imprimir_tablero(tab)

    algos = {'1': ("DFS", dfs), '2': ("BFS", bfs), '3': ("UCS", ucs), '4': ("A*", a_estrella)}
    print("1. DFS  |  2. BFS  |  3. UCS  |  4. A*  |  5. Comparar los 4")
    op = input("Elige opción: ").strip()

    if op in algos:
        nom, fn = algos[op]
        cols, costo, exp, t = fn(tab)
        print(f"\n[{nom}] Columnas jugadas: {cols} | Costo total: {costo} | Nodos expandidos: {exp} | Tiempo: {t*1000:.3f} ms")
    elif op == '5':
        print(f"\n{'Algoritmo':<10} | {'Jugadas':<12} | {'Costo Total':<12} | {'Nodos Exp.':<10} | {'Tiempo (ms)':<10}")
        print("-" * 65)
        for _, (nom, fn) in algos.items():
            cols, costo, exp, t = fn(tab)
            print(f"{nom:<10} | {str(cols):<12} | {costo:<12.2f} | {exp:<10} | {t*1000:<10.3f}")

def menu_versus():
    tab = crear_tablero()
    print("\n--- MODO DUELO VS IA (MINIMAX CON PODA) ---")
    imprimir_tablero(tab)

    while True:
        # Humano (X)
        try:
            col = int(input("Tu jugada (columna 0-6): "))
            nuevo = soltar_ficha(tab, col, 'X')
            if nuevo is None:
                print("¡Columna llena! Elige otra.")
                continue
            tab = nuevo
        except (ValueError, IndexError):
            print("Entrada inválida.")
            continue

        imprimir_tablero(tab)
        if es_meta(tab, 'X'):
            print("🎉 ¡Has ganado!"); break
        if all(tab[0][c] != ' ' for c in range(COLUMNAS)):
            print("🤝 ¡Empate!"); break

        # Máquina (O)
        print("La IA está calculando su jugada...")
        col_ia, con_poda, sin_poda = mejor_columna_ia(tab, prof=4)
        tab = soltar_ficha(tab, col_ia, 'O')
        imprimir_tablero(tab)

        print(f"📊 MÉTRICAS MINIMAX:")
        print(f"  • Jugada elegida: Columna {col_ia}")
        print(f"  • Nodos evaluados CON poda: {con_poda}")
        print(f"  • Nodos evaluados SIN poda: {sin_poda}")
        ahorro = ((sin_poda - con_poda) / sin_poda * 100) if sin_poda > 0 else 0
        print(f"  • Ahorro de búsqueda: {ahorro:.1f}%\n")

        if es_meta(tab, 'O'):
            print("🤖 ¡La IA ha ganado!"); break
        if all(tab[0][c] != ' ' for c in range(COLUMNAS)):
            print("🤝 ¡Empate!"); break

def main():
    while True:
        print("\n========================================")
        print("   PROYECTO 1: CONECTA 4 INTELIGENTE")
        print("========================================")
        print("1. Modo Puzzle / Desafío (DFS, BFS, UCS, A*)")
        print("2. Modo Duelo vs Máquina (Minimax Alfa-Beta)")
        print("3. Salir")
        op = input("Opción: ").strip()
        if op == '1': menu_puzzle()
        elif op == '2': menu_versus()
        elif op == '3': break

if __name__ == "__main__":
    main()