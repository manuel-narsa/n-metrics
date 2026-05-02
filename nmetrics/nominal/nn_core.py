import pandas as pd
import numpy as np
import math
import itertools
from collections import defaultdict
from scipy.special import gammaln

# ==============================================================================
# 1. CONSTRUCCIÓN DEL HIPERESPACIO Y MACROESTADOS (NIVEL III)
# ==============================================================================
def _build_macrostate_dictionary_nn(m_jueces, k_escala):
    formas = list(itertools.combinations_with_replacement(range(1, k_escala + 1), m_jueces))
    macro_dict = defaultdict(float)
    max_coincidencias = (m_jueces * (m_jueces - 1)) / 2
    
    for forma in formas:
        forma_arr = np.array(forma)
        counts = [np.sum(forma_arr == val) for val in range(1, k_escala + 1)]
        
        coincidencias = sum(c * (c - 1) / 2 for c in counts)
        acuerdo = round(coincidencias / max_coincidencias, 8) if max_coincidencias > 0 else 0.0
        
        # 1. Permutaciones de jueces: m! / (c1! c2! ...)
        denom_jueces = 1.0
        for c in counts: 
            denom_jueces *= math.factorial(c)
        perm_jueces = math.factorial(m_jueces) / denom_jueces
        
        # ¡CRÍTICO! No se multiplica por las permutaciones de categorías (perm_cats)
        # porque itertools.combinations_with_replacement ya recorre todas las combinaciones 
        # posibles de categorías implícitamente. Multiplicar aquí causaría sobrecuento.
        
        macro_dict[acuerdo] += perm_jueces
        
    return macro_dict

# ==============================================================================
# 2. MOTOR TERMODINÁMICO DE SIMULACIÓN PONDERADA (IPW)
# ==============================================================================
def _calcular_pesos_termodinamicos_sujetos_nn(X_3d, k_escala):
    S, n, m = X_3d.shape
    counts = np.zeros((S, n, k_escala))
    
    # Limpiamos los datos y contamos frecuencias
    X_cat = np.floor(X_3d + 0.5)
    X_cat = np.clip(X_cat, 1, k_escala)
    
    for k_val in range(1, k_escala + 1):
        counts[:, :, k_val-1] = np.sum(X_cat == k_val, axis=2)
        
    m_valid = np.sum(~np.isnan(X_3d), axis=2)
    
    # 1. Permutaciones de los jueces: m! / (c1! c2! ...)
    log_m_fact = gammaln(m_valid + 1)
    log_denom_jueces = np.sum(gammaln(counts + 1), axis=2)
    log_perm_jueces = log_m_fact - log_denom_jueces
    
    # 2. Asignación de categorías: k! / (f0! f1! ... fm!)
    f_c = np.zeros((S, n, m + 1))
    for c in range(m + 1):
        f_c[:, :, c] = np.sum(counts == c, axis=2)
        
    log_k_fact = gammaln(k_escala + 1)
    log_denom_cat = np.sum(gammaln(f_c + 1), axis=2)
    log_asign_cat = log_k_fact - log_denom_cat
    
    # 3. Multiplicidad Termodinámica Teórica
    log_omega = log_perm_jueces + log_asign_cat
    max_log = np.max(log_omega, axis=1, keepdims=True) # Previene desbordamiento
    omega_teorico = np.exp(log_omega - max_log)
    
    # 4. CORRECCIÓN IPW (Frecuencia Empírica de Filas)
    counts_sorted = np.sort(counts, axis=2)
    
    f_emp = np.ones((S, n))
    for s in range(S):
        _, inverse_idx, freq = np.unique(counts_sorted[s], axis=0, return_inverse=True, return_counts=True)
        f_emp[s] = freq[inverse_idx]
        
    # Ponderación Inversa (IPW)
    omega_corregido = omega_teorico / f_emp
    
    # Normalización al 100%
    sum_omega = np.sum(omega_corregido, axis=1, keepdims=True)
    pesos = np.where(sum_omega > 0, omega_corregido / sum_omega, 1.0 / n)
    
    return pesos

def _compute_nn_sp_vectorized(X_3d, w_pesos_filas, k_escala):
    S, n, m = X_3d.shape
    m_valid = np.sum(~np.isnan(X_3d), axis=2)
    
    counts = np.zeros((S, n, k_escala))
    X_clip = np.floor(X_3d + 0.5)
    X_clip = np.clip(X_clip, 1, k_escala)
    for k_val in range(1, k_escala + 1):
        counts[:, :, k_val-1] = np.sum(X_clip == k_val, axis=2)
        
    coincidencias_row = np.sum(counts * (counts - 1) / 2, axis=2)
    max_coincidencias = m_valid * (m_valid - 1) / 2
    
    acuerdo_row = np.divide(coincidencias_row, max_coincidencias,
                            out=np.full_like(coincidencias_row, np.nan, dtype=float),
                            where=max_coincidencias > 0)
    
    valid_rows = ~np.isnan(acuerdo_row) & (m_valid > 1)
    pesos_validos = np.where(valid_rows, w_pesos_filas, 0.0)
    
    sum_pesos = np.sum(pesos_validos, axis=1, keepdims=True)
    pesos_norm = np.where(sum_pesos > 0, pesos_validos / sum_pesos, 0.0)
    
    mu_global = np.sum(acuerdo_row * pesos_norm, axis=1)
    var_global = np.sum(pesos_norm * (acuerdo_row - mu_global[:, None])**2, axis=1)
    sigma_global = np.sqrt(var_global)
    
    return np.sqrt(np.maximum(mu_global * (1 - sigma_global), 0.0))

# ==============================================================================
# 3. INTERFAZ ESTADÍSTICA PRINCIPAL
# ==============================================================================
def calcular_estadisticas_nn(matriz_entrada, S_replicas, k_escala=5, metodo_ic='SP'):
    X = np.array(matriz_entrada, dtype=float)
    n, m = X.shape
    
    # Acuerdo muestral (Pesos planos)
    w_plano = np.ones((1, n)) / n 
    nn_muestra = _compute_nn_sp_vectorized(X[None, :, :], w_plano, k_escala)[0]
    
    if metodo_ic == 'SP':
        # 1. Calculamos la ponderación termodinámica (IPW)
        w_termo = _calcular_pesos_termodinamicos_sujetos_nn(X[None, :, :], k_escala)
        p_i = w_termo[0] 
        
        # 2. Calculamos el acuerdo poblacional proyectado
        nn_ponderado = _compute_nn_sp_vectorized(X[None, :, :], p_i[None, :], k_escala)[0]
        
        # 3. Simulación Bootstrap guiada por IPW
        indices = np.random.choice(n, size=(S_replicas, n), replace=True, p=p_i)
        X_replicas = X[indices]
        w_flat_replicas = np.ones((S_replicas, n)) / n 
        nn_replicas = _compute_nn_sp_vectorized(X_replicas, w_flat_replicas, k_escala)
        
    else:
        nn_ponderado = nn_muestra # Fallback si BC clásico
        indices = np.random.choice(n, size=(S_replicas, n), replace=True)
        X_replicas = X[indices]
        w_flat_replicas = np.ones((S_replicas, n)) / n
        nn_replicas = _compute_nn_sp_vectorized(X_replicas, w_flat_replicas, k_escala)
    
    mascara_validos = ~np.isnan(nn_replicas)
    nn_replicas_valid = nn_replicas[mascara_validos]
    X_replicas_valid = X_replicas[mascara_validos]
    
    if len(nn_replicas_valid) < 2: 
        return float(nn_muestra), float(nn_ponderado), np.nan, np.nan, nn_replicas_valid, X_replicas_valid
    return float(nn_muestra), float(nn_ponderado), float(np.percentile(nn_replicas_valid, 2.5)), float(np.percentile(nn_replicas_valid, 97.5)), nn_replicas_valid, X_replicas_valid

def detectar_anomalias_nn(matriz_entrada, k_escala=5, umbral_sigma=1.0):
    X = np.array(matriz_entrada, dtype=float)
    m_valid = np.sum(~np.isnan(X), axis=1)
    
    counts = np.zeros((len(X), k_escala))
    X_clip = np.floor(X + 0.5); X_clip = np.clip(X_clip, 1, k_escala)
    for k_val in range(1, k_escala + 1):
        counts[:, k_val-1] = np.sum(X_clip == k_val, axis=1)
        
    coincidencias_row = np.sum(counts * (counts - 1) / 2, axis=1)
    max_coincidencias = m_valid * (m_valid - 1) / 2
    
    acuerdo_local = np.divide(coincidencias_row, max_coincidencias,
                              out=np.full_like(coincidencias_row, np.nan, dtype=float),
                              where=max_coincidencias > 0)
    
    valid = ~np.isnan(acuerdo_local)
    if not np.any(valid): return pd.DataFrame(), np.nan, np.nan, np.nan
    
    mu_global = np.mean(acuerdo_local[valid])
    sigma_global = np.std(acuerdo_local[valid])
    limite = mu_global - (umbral_sigma * sigma_global)
    anomalo = (acuerdo_local < limite) & valid
    df_anomalias = pd.DataFrame({'Sujeto_ID': np.arange(1, len(X) + 1), 'Acuerdo_Local': acuerdo_local, 'Es_Anomalo': anomalo})
    return df_anomalias, mu_global, sigma_global, limite

def calcular_azar_termodinamico_nn(m_jueces, k_escala=5):
    macro_dict = _build_macrostate_dictionary_nn(m_jueces, k_escala)
    acuerdos = np.array(list(macro_dict.keys()))
    w_topologico = np.array(list(macro_dict.values()))
    
    sum_w = np.sum(w_topologico)
    mu_global = np.sum(w_topologico * acuerdos) / sum_w
    var_global = np.sum(w_topologico * (acuerdos - mu_global)**2) / sum_w
    sigma_global = np.sqrt(var_global)
    
    return np.sqrt(max(0.0, mu_global * (1.0 - sigma_global)))

def calcular_percentil_universal_nn(nn_empirico, m_jueces, k_escala=5):
    macro_dict = _build_macrostate_dictionary_nn(m_jueces, k_escala)
    agrupados = defaultdict(float)
    total_espacio = sum(macro_dict.values()) 
    
    for acuerdo, multiplicidad_ce in macro_dict.items():
        nn_val = np.sqrt(acuerdo)
        agrupados[round(nn_val, 6)] += multiplicidad_ce
        
    ordenados = sorted(agrupados.items())
    percentiles = []
    acumulado = 0.0
    for nn_val, mult in ordenados:
        acumulado += mult
        percentil_sup = acumulado / total_espacio
        percentiles.append({'nn': nn_val, 'p_sup': percentil_sup})
        
    nn_lower = None; p_lower = None
    nn_upper = None; p_upper = None
    
    for p in percentiles:
        if p['nn'] <= nn_empirico:
            nn_lower = p['nn']; p_lower = p['p_sup']
    for p in reversed(percentiles):
        if p['nn'] >= nn_empirico:
            nn_upper = p['nn']; p_upper = p['p_sup']
            
    if nn_lower is None: return 0.0
    if nn_upper is None: return 100.0
    if nn_lower == nn_upper: return p_lower * 100.0
    
    p_interp = p_lower + (p_upper - p_lower) * (nn_empirico - nn_lower) / (nn_upper - nn_lower)
    return p_interp * 100.0