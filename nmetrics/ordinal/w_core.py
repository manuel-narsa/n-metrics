import numpy as np
import pandas as pd
from math import comb

# ==============================================================================
# REEMPLAZO EN w_core.py: KENDALL'S W INMUNE A VALORES ABSOLUTOS
# ==============================================================================
# ==============================================================================
# REEMPLAZO EN w_core.py: KENDALL'S W OPTIMIZADO (RESISTENTE A BIG DATA)
# ==============================================================================
# ==============================================================================
# REEMPLAZO EN w_core.py: KENDALL'S W CON OPCIÓN DE CORRECCIÓN DE EMPATES
# ==============================================================================
def _compute_w_vectorized(X_3d, ajustar_empates=True):
    """
    Cálculo de Kendall's W.
    - ajustar_empates=True:  Fórmula exacta de Kendall (1970) / Siegel (0.1436).
    - ajustar_empates=False: Fórmula clásica sin corregir empates / PQStat (0.1379).
    """
    S, n, m = X_3d.shape
    
    if ajustar_empates:
        unique_vals = np.unique(X_3d[~np.isnan(X_3d)])
        k_real = len(unique_vals)
        counts = np.zeros((S, m, k_real))
        
        for idx, val in enumerate(unique_vals):
            counts[:, :, idx] = np.sum(X_3d == val, axis=1)
            
        T_j = np.sum(counts**3 - counts, axis=2)
        T_total = np.sum(T_j, axis=1)
        denom = (m**2) * (n**3 - n) - m * T_total
    else:
        denom = (m**2) * (n**3 - n)
    
    # Suma de Rangos (S_val)
    S_val = np.zeros(S)
    mean_R = m * (n + 1) / 2.0
    
    for s in range(S):
        X_s = X_3d[s]
        ranks = pd.DataFrame(X_s).rank(method='average').values
        R_i = np.sum(ranks, axis=1)
        S_val[s] = np.sum((R_i - mean_R)**2)
        
    with np.errstate(divide='ignore', invalid='ignore'):
        W = np.where(denom == 0, np.nan, 12.0 * S_val / denom)
        
    return W

# ==============================================================================
# 2. Cálculo Teórico Poblacional (Matriz Asintótica con M_pop Adaptativo)
# ==============================================================================
def calcular_w_poblacion_asintotica(matriz_entrada, k_escala=5, multiplicador=None):
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