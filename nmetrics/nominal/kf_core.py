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
        
        # 1. Permutaciones de jueces
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
# REEMPLAZO EN kf_core.py: CONTEO AGNÓSTICO (INMUNE A ETIQUETAS ABSOLUTAS)
# ==============================================================================
def _compute_kf_vectorized_bc(X_3d, k_escala=5):
    """Calcula Fleiss Kappa asumiendo equiprobabilidad estricta y conteo dinámico."""
    S, n, m = X_3d.shape
    
    # Extraemos solo las categorías que realmente existen en el dataset
    unique_vals = np.unique(X_3d[~np.isnan(X_3d)])
    
    # Frecuencias por categoría existente
    counts = np.zeros((S, n, len(unique_vals)))
    for idx, val in enumerate(unique_vals):
        counts[:, :, idx] = np.sum(X_3d == val, axis=2)
        
    # Acuerdo Observado (Po) por sujeto
    po_subj = np.sum(counts * (counts - 1), axis=2) / (m * (m - 1)) # (S, n)
    Po = np.mean(po_subj, axis=1) # (S,)
    
    # Acuerdo Esperado (Pe) basado solo en las marginales activas
    total_counts = np.sum(counts, axis=1) # (S, len(unique_vals))
    Pe = np.sum((total_counts / (n * m)) ** 2, axis=1) # (S,)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        kf_vals = np.where(Pe == 1, np.nan, (Po - Pe) / (1 - Pe))
        
    return kf_vals

# ==============================================================================
# 3. Cálculo Teórico Poblacional (Matriz Asintótica Masiva)
# ==============================================================================
def calcular_kf_poblacion_asintotica(matriz_entrada, k_escala=5, multiplicador=1000):
    """
    Construye el universo termodinámico físico clonando filas y lo evalúa
    usando la fórmula de Kappa de Fleiss original pura.
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
    
    # Evaluar el universo entero con la función pura de KF
    X_massive_3d = X_massive[None, :, :]
    kf_pob_real = _compute_kf_vectorized_bc(X_massive_3d, k_escala)[0]
    
    return float(kf_pob_real)

# ==============================================================================
# 4. Cálculo del Coeficiente Muestral y su IC (Alineación Estricta PQStat Fleiss, 1971)
# ==============================================================================
def calcular_estadisticas_kf(matriz_entrada, S_replicas, k_escala=5):
    """
    Calcula Fleiss Kappa y su Intervalo de Confianza utilizando la 
    Varianza Asintótica No-Nula exacta de Fleiss, Cohen & Everitt (1979).
    Esta es la fórmula interna que utiliza PQStat y SPSS para los ICs.
    """
    matriz_empirica = np.array(matriz_entrada, dtype=float)
    n, m = matriz_empirica.shape
    
    # 1. Extracción topológica agnóstica
    unique_vals = np.unique(matriz_empirica[~np.isnan(matriz_empirica)])
    
    counts = np.zeros((n, len(unique_vals)))
    for idx, val in enumerate(unique_vals):
        counts[:, idx] = np.sum(matriz_empirica == val, axis=1)
        
    # 2. Probabilidades Marginales Globales (p_j) y (q_j)
    total_counts = np.sum(counts, axis=0)
    pj = total_counts / (n * m)
    qj = 1.0 - pj
    
    # Acuerdo Esperado (Pe)
    Pe = np.sum(pj ** 2)
    
    # 3. Acuerdo Observado (Po)
    A_i = np.sum(counts * (counts - 1), axis=1) / (m * (m - 1))
    Po = np.mean(A_i)
    
    # Protección matemática contra infinitos
    if Pe == 1.0:
        return {'KF Muestra': np.nan, 'IC Inf': np.nan, 'IC Sup': np.nan}
        
    # 4. Fleiss Kappa Muestral
    kf_muestra = (Po - Pe) / (1 - Pe)
    
    # 5. Varianza de Fleiss, Cohen & Everitt (1979)
    # PQStat utiliza esta formulación algebraica en lugar del Método Delta lineal.
    sum_pj_qj = np.sum(pj * qj)
    
    term_1 = sum_pj_qj ** 2
    term_2 = np.sum(pj * qj * (qj - pj))
    
    var_k = (2.0 / (n * m * (m - 1))) * ((term_1 - term_2) / (sum_pj_qj ** 2))
    ase = np.sqrt(max(0, var_k))
    
    # 6. Construcción del IC Clásico de PQStat (Z = 1.96)
    ic_inf = kf_muestra - 1.96 * ase
    ic_sup = kf_muestra + 1.96 * ase
    
    # Acotamiento superior
    ic_sup = min(1.0, ic_sup)
    
    return {
        'KF Muestra': kf_muestra,
        'IC Inf': ic_inf,
        'IC Sup': ic_sup
    }