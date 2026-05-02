import numpy as np
import pandas as pd
from math import comb

# ==============================================================================
# 1. Motor Vectorizado de Kendall W Clásico
# ==============================================================================
def _compute_w_vectorized(X_3d):
    """
    Cálculo vectorizado de Kendall's W (Coeficiente de Concordancia Ordinal)
    """
    S, n, m = X_3d.shape
    diff = X_3d[:, :, None, :] - X_3d[:, None, :, :] 
    less = (diff < 0).astype(float)
    equal = (diff == 0).astype(float)
    
    ranks = np.sum(less + 0.5 * equal, axis=2) + 0.5 
    R_i = np.sum(ranks, axis=2) 
    mean_R = m * (n + 1) / 2.0
    S_val = np.sum((R_i - mean_R)**2, axis=1) 
    
    k_escala = int(np.nanmax(X_3d))
    counts = np.zeros((S, m, k_escala))
    for k in range(1, k_escala + 1):
        counts[:, :, k-1] = np.sum(X_3d == k, axis=1)
        
    T_j = np.sum(counts**3 - counts, axis=2) 
    T_total = np.sum(T_j, axis=1) 
    
    denom = (m**2) * (n**3 - n) - m * T_total
    #W = np.where(denom == 0, np.nan, (12 * S_val) / denom)
    W = np.divide(12 * S_val, denom, 
              out=np.full_like(denom, np.nan, dtype=float), 
              where=(denom != 0))
    return W

# ==============================================================================
# 2. Cálculo Teórico Poblacional (Matriz Asintótica Masiva de Columnas)
# ==============================================================================
def calcular_w_poblacion_asintotica(matriz_entrada, k_escala=5, multiplicador=1000):
    n, m = matriz_entrada.shape
    c_arr = np.zeros(m, dtype=int)
    for val in range(1, k_escala + 1):
        c_arr += (matriz_entrada == val).any(axis=0).astype(int)
        
    pesos = np.array([comb(k_escala, c) for c in c_arr], dtype=float)
    if np.sum(pesos) == 0: return np.nan
    prob = pesos / np.sum(pesos)
    
    M_pop = m * multiplicador
    counts = np.round(prob * M_pop).astype(int)
    
    diff = M_pop - np.sum(counts)
    if diff > 0: counts[np.argmax(prob)] += diff
    elif diff < 0: counts[np.argmax(counts)] += diff
        
    X_massive = np.repeat(matriz_entrada, counts, axis=1)
    w_pob_real = _compute_w_vectorized(X_massive[None, :, :])[0]
    return float(w_pob_real)

# ==============================================================================
# 3. Cálculo del Coeficiente Muestral y su IC
# ==============================================================================
def calcular_estadisticas_w(matriz_entrada, S_replicas, k_escala=5):
    matriz_empirica = np.array(matriz_entrada, dtype=float)
    n_sujetos, m_evaluadores = matriz_empirica.shape
    
    X_muestra_3d = matriz_empirica[None, :, :]
    w_muestra = _compute_w_vectorized(X_muestra_3d)[0]
    
    # Bootstrap Clásico sobre filas (sujetos)
    indices = np.empty((S_replicas, n_sujetos), dtype=int)
    for s in range(S_replicas):
        indices[s] = np.random.choice(n_sujetos, size=n_sujetos, replace=True)
        
    X_3d_bc = matriz_empirica[indices]
    w_replicas = _compute_w_vectorized(X_3d_bc)
    
    clean_w = w_replicas[~np.isnan(w_replicas)]
    if len(clean_w) > 0:
        ic_inf, ic_sup = np.percentile(clean_w, [2.5, 97.5])
    else:
        ic_inf, ic_sup = np.nan, np.nan
        
    return {
        'W Muestra': w_muestra,
        'IC Inf': ic_inf,
        'IC Sup': ic_sup
    }