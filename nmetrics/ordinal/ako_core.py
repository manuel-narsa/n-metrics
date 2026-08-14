import numpy as np
import pandas as pd
from math import comb

# ==============================================================================
# REEMPLAZO EN ako_core.py: CONTEO AGNÓSTICO PARA ALFA ORDINAL
# ==============================================================================
def _compute_ako_vectorized(X_3d, k_escala=None):
    """Calcula el Alpha Ordinal de Krippendorff usando categorías dinámicas."""
    S, n, m = X_3d.shape
    N_total = n * m
    
    # Extraemos solo los valores únicos existentes (ordenados de menor a mayor)
    unique_vals = np.sort(np.unique(X_3d[~np.isnan(X_3d)]))
    k = len(unique_vals)
    
    freqs = np.zeros((S, k))
    row_freqs = np.zeros((S, n, k))
    
    for v, val in enumerate(unique_vals):
        mask = (X_3d == val)
        freqs[:, v] = np.sum(mask, axis=(1, 2))
        row_freqs[:, :, v] = np.sum(mask, axis=2)

    D = np.zeros((S, k, k))
    for c in range(k):
        for k_val in range(c + 1, k):
            # La suma de frecuencias ignora los ceros de forma nativa
            s_val = np.sum(freqs[:, c:k_val+1], axis=1)
            d = s_val - freqs[:, c]/2.0 - freqs[:, k_val]/2.0
            D[:, c, k_val] = d**2
            D[:, k_val, c] = d**2

    De = np.einsum('si,sj,sij->s', freqs, freqs, D)
    Do = np.einsum('sni,snj,sij->s', row_freqs, row_freqs, D) / (m - 1.0)

    D_OO = Do * (N_total - 1.0)
    D_EO = De

    with np.errstate(divide='ignore', invalid='ignore'):
        ako = np.where(D_EO == 0, np.nan, 1.0 - (D_OO / D_EO))
        
    return ako

# ==============================================================================
# 2. Cálculo Teórico Poblacional (Matriz Asintótica con M_pop Adaptativo)
# ==============================================================================
def calcular_ako_poblacion_asintotica(matriz_entrada, k_escala=5, multiplicador=None):
    n, m = matriz_entrada.shape
    c_arr = np.zeros(m, dtype=int)
    for val in range(1, k_escala + 1):
        c_arr += (matriz_entrada == val).any(axis=0).astype(int)
        
    pesos = np.array([comb(k_escala, c) for c in c_arr], dtype=float)
    sum_pesos = np.sum(pesos)
    if sum_pesos == 0: 
        return np.nan
        
    prob = pesos / sum_pesos
    
    # --- 1. ESCALADO ADAPTATIVO DE EVALUADORES (COLUMNAS) ---
    if multiplicador is None:
        # Acotamos M_pop entre 100 y 1.000 columnas para proteger la RAM
        M_pop = int(np.clip(m * 100, 100, 1_000))
    else:
        M_pop = int(m * multiplicador)
    
    counts = np.round(prob * M_pop).astype(int)
    
    # Ajuste de diferencias por redondeo
    diff = M_pop - np.sum(counts)
    if diff > 0: 
        counts[np.argmax(prob)] += diff
    elif diff < 0: 
        counts[np.argmax(counts)] += diff
        
    # --- 2. SANITIZACIÓN CRÍTICA CONTRA VALORES NEGATIVOS ---
    counts = np.maximum(0, counts).astype(int)
        
    X_massive = np.repeat(matriz_entrada, counts, axis=1)
    ako_pob_real = _compute_ako_vectorized(X_massive[None, :, :], k_escala)[0]
    return float(ako_pob_real)

# ==============================================================================
# 3. Cálculo del Coeficiente Muestral y su IC
# ==============================================================================
def calcular_estadisticas_ako(matriz_entrada, S_replicas, k_escala=5):
    matriz_empirica = np.array(matriz_entrada, dtype=float)
    n_sujetos, m_evaluadores = matriz_empirica.shape
    
    X_muestra_3d = matriz_empirica[None, :, :]
    ako_muestra = _compute_ako_vectorized(X_muestra_3d, k_escala)[0]
    
    # Bootstrap Clásico sobre filas (sujetos)
    indices = np.empty((S_replicas, n_sujetos), dtype=int)
    for s in range(S_replicas):
        indices[s] = np.random.choice(n_sujetos, size=n_sujetos, replace=True)
        
    X_3d_bc = matriz_empirica[indices]
    ako_replicas = _compute_ako_vectorized(X_3d_bc, k_escala)
    
    clean_ako = ako_replicas[~np.isnan(ako_replicas)]
    if len(clean_ako) > 0:
        ic_inf, ic_sup = np.percentile(clean_ako, [2.5, 97.5])
    else:
        ic_inf, ic_sup = np.nan, np.nan
        
    return {
        'AKO Muestra': ako_muestra,
        'IC Inf': ic_inf,
        'IC Sup': ic_sup
    }
