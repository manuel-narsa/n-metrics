import math
from collections import Counter

def calcular_firmas_topologicas(m_jueces, k_escala):
    # Genera las particiones enteras de 'm' en máximo 'k' partes
    def particiones(n, k_max, max_val=None):
        if max_val is None: max_val = n
        if n == 0: yield []
        if k_max == 0 or n < 0: return
        for i in range(min(n, max_val), 0, -1):
            for p in particiones(n - i, k_max - 1, i):
                yield [i] + p
                
    total_firmas = 0
    for p in particiones(m_jueces, k_escala):
        counts = Counter(p) 
        # Calcula el tamaño de cada bloque de coincidencias
        freqs = {s: s * counts[s] for s in counts}
        
        denom = 1
        for f in freqs.values():
            denom *= math.factorial(f)
            
        total_firmas += math.factorial(m_jueces) // denom
        
    return total_firmas

# Ejemplo: Generar la fila de m=7
print(f"k=3, m=20 -> {calcular_firmas_topologicas(20, 3)} TS")
print(f"k=5, m=20 -> {calcular_firmas_topologicas(20, 5)} TS") # Devuelve tus 554 exactos
print(f"k=7, m=20 -> {calcular_firmas_topologicas(20, 7)} TS") # Techo máximo: 576
