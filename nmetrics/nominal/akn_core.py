import pandas as pd
import numpy as np
from math import factorial
from collections import Counter

# ==============================================================================
# 1. Preparar población teórica (Espacio de Configuración Nominal)
# ==============================================================================
def calcular_parametros_poblacion_nominal(matriz_entrada, k_escala):
    n, m = matriz_entrada.shape
    filas_salida = []
    fact_m = factorial(m)
    fact_k = factorial(k_escala)
    
    for fila in matriz_entrada:
        df_counts = [np.sum(fila == v) for v in range(1, k_escala + 1)]
        
        # 1. Permutaciones de jueces (Tuplas de Evaluadores)
        denominador_m = 1
        for c in df_counts:
            denominador_m *= factorial(int(c))
        tuplas_evals = fact_m // denominador_m
        
        # 2. Permutaciones de la escala (Invarianza categórica)
        counts_of_counts = Counter(df_counts)
        denominador_k = 1
        for freq in counts_of_counts.values():
            denominador_k *= factorial(freq)
        df_val = fact_k // denominador_k
        
        # Volumen termodinámico de este patrón nominal
        num_tuplas = df_val * tuplas_evals
        
        filas_salida.append(list(fila) + [num_tuplas])
        
    cols_x = [f"X{i+1}" for i in range(m)]
    columnas = cols_x + ["Num_Tuplas"]
    return pd.DataFrame(filas_salida, columns=columnas), cols_x

# ==============================================================================
# 2. Motor Vectorizado de AKN Clásico (Plano / Sin Pesos)
# ==============================================================================
def _compute_akn_vectorized_bc(X_3d, k_escala=5):
    """Calcula AKN asumiendo equiprobabilidad estricta (1/n)."""
    S, n, m = X_3d.shape
    
    # Frecuencias por categoría: (S, n, k_escala)
    counts = np.zeros((S, n, k_escala))
    for k in range(1, k_escala + 1):
        counts[:, :, k-1] = np.sum(X_3d == k, axis=2)
        
    # StoreAgreement por sujeto: sum_c (n_ic * (n_ic - 1)) / (m * (m - 1))
    store_agreement = np.sum(counts * (counts - 1), axis=2) / (m * (m - 1)) # (S, n)
    
    mean_store_agreement = np.mean(store_agreement, axis=1) # (S,)
    pa = (1 - 1 / (n * m)) * mean_store_agreement + 1 / (n * m) # (S,)
    
    # Proporciones por categoría
    p_c = np.sum(counts, axis=1) / (n * m) # (S, k_escala)
    pe = np.sum(p_c ** 2, axis=1) # (S,)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        akn_vals = np.where(pe == 1, np.nan, (pa - pe) / (1 - pe))
        
    return akn_vals

# ==============================================================================
# 3. Cálculo Teórico Poblacional (Matriz Asintótica Masiva)
# ==============================================================================
def calcular_akn_poblacion_asintotica(matriz_entrada, k_escala=5, multiplicador=1000):
    """
    Construye el universo termodinámico físico clonando filas y lo evalúa
    usando la fórmula de AKN original pura.
    """
    df_pob, cols_x = calcular_parametros_poblacion_nominal(matriz_entrada, k_escala)
    pesos = df_pob['Num_Tuplas'].values.astype(float)
    prob = pesos / np.sum(pesos)
    
    n, k = matriz_entrada.shape
    N_pop = n * multiplicador
    
    counts = np.round(prob * N_pop).astype(int)
    
    diff = N_pop - np.sum(counts)
    if diff > 0: counts[np.argmax(prob)] += diff
    elif diff < 0: counts[np.argmax(counts)] += diff
        
    X_massive = np.repeat(matriz_entrada, counts, axis=0)
    
    # Evaluar el universo entero con la función pura de AKN
    X_massive_3d = X_massive[None, :, :]
    akn_pob_real = _compute_akn_vectorized_bc(X_massive_3d, k_escala)[0]
    
    return float(akn_pob_real)

# ==============================================================================
# 4. Cálculo del Coeficiente Muestral y su IC
# ==============================================================================
def calcular_estadisticas_akn(matriz_entrada, S_replicas, k_escala=5):
    matriz_empirica = np.array(matriz_entrada, dtype=float)
    n_sujetos, m_evaluadores = matriz_empirica.shape
    
    # 1. Cálculo puntual de AKN para la matriz de muestra
    X_muestra_3d = matriz_empirica[None, :, :]
    akn_muestra = _compute_akn_vectorized_bc(X_muestra_3d, k_escala)[0]
    
    # 2. Motor de Caos: Bootstrap Clásico (Probabilidad 1/n)
    indices = np.empty((S_replicas, n_sujetos), dtype=int)
    for s in range(S_replicas):
        indices[s] = np.random.choice(n_sujetos, size=n_sujetos, replace=True)
        
    X_3d_bc = matriz_empirica[indices]
    
    # 3. Evaluación vectorizada de réplicas
    akn_replicas = _compute_akn_vectorized_bc(X_3d_bc, k_escala)
    
    # 4. Cálculo de los percentiles (IC)
    clean_akn = akn_replicas[~np.isnan(akn_replicas)]
    if len(clean_akn) > 0:
        ic_inf, ic_sup = np.percentile(clean_akn, [2.5, 97.5])
    else:
        ic_inf, ic_sup = np.nan, np.nan
        
    return {
        'AKN Muestra': akn_muestra,
        'IC Inf': ic_inf,
        'IC Sup': ic_sup
    }