import pandas as pd
import numpy as np
import math
import itertools
from collections import defaultdict
from scipy.special import gammaln

def _build_macrostate_dictionary(m_jueces, k_escala):
    formas = list(itertools.combinations_with_replacement(range(1, k_escala + 1), m_jueces))
    
    n_ext1 = m_jueces // 2
    n_ext2 = m_jueces - n_ext1
    mean_ext = (n_ext1 * 1 + n_ext2 * k_escala) / m_jueces
    var_ext = (n_ext1 * (1 - mean_ext)**2 + n_ext2 * (k_escala - mean_ext)**2) / m_jueces
    max_sigma = np.sqrt(var_ext)
    
    macro_dict = defaultdict(float)
    
    for forma in formas:
        forma_arr = np.array(forma)
        sigma = np.std(forma_arr)
        acuerdo = max(0.0, 1.0 - (sigma / max_sigma))
        
        acuerdo_key = round(acuerdo, 8)
        
        counts = [np.sum(forma_arr == k) for k in range(1, k_escala + 1)]
        denom = 1.0
        for c in counts: denom *= math.gamma(c + 1)
        multiplicidad_fc = math.gamma(m_jueces + 1) / denom
        
        macro_dict[acuerdo_key] += multiplicidad_fc
        
    return macro_dict

def _compute_ni_sp_vectorized(X_3d, w_pesos_filas, k_escala):
    S, n, m = X_3d.shape
    m_valid = np.sum(~np.isnan(X_3d), axis=2) 
    
    n_ext1 = m_valid // 2
    n_ext2 = m_valid - n_ext1
    safe_m = np.where(m_valid == 0, 1, m_valid)
    mean_ext = (n_ext1 * 1 + n_ext2 * k_escala) / safe_m
    var_ext = (n_ext1 * (1 - mean_ext)**2 + n_ext2 * (k_escala - mean_ext)**2) / safe_m
    max_sigma_row = np.where(m_valid > 1, np.sqrt(var_ext), np.nan) 
    
    sigma_row = np.nanstd(X_3d, axis=2)
    acuerdo_row = np.where(max_sigma_row > 0, 1.0 - (sigma_row / max_sigma_row), np.nan)
    
    valid_rows = ~np.isnan(acuerdo_row) & (m_valid > 1)
    pesos_validos = np.where(valid_rows, w_pesos_filas, 0.0)
    
    sum_pesos = np.sum(pesos_validos, axis=1, keepdims=True)
    pesos_norm = np.where(sum_pesos > 0, pesos_validos / sum_pesos, 0.0)
    
    mu_global = np.sum(acuerdo_row * pesos_norm, axis=1)
    var_global = np.sum(pesos_norm * (acuerdo_row - mu_global[:, None])**2, axis=1)
    sigma_global = np.sqrt(var_global)
    
    return np.sqrt(np.maximum(mu_global * (1 - sigma_global), 0.0))

def calcular_estadisticas_ni(matriz_entrada, S_replicas, k_escala=5, metodo_ic='SP'):
    X = np.array(matriz_entrada, dtype=float)
    n, m = X.shape
    
    w_plano = np.ones((1, n)) / n 
    ni_muestra = _compute_ni_sp_vectorized(X[None, :, :], w_plano, k_escala)[0]
    
    ni_ponderado = ni_muestra 
    
    if metodo_ic == 'SP':
        counts = np.zeros((n, k_escala))
        m_valid = np.sum(~np.isnan(X), axis=1)
        
        for k_val in range(1, k_escala + 1):
            distancia = np.abs(X - k_val)
            masa_difusa = np.maximum(0.0, 1.0 - distancia)
            counts[:, k_val-1] = np.nansum(masa_difusa, axis=1)
            
        log_omega = gammaln(m_valid + 1) - np.sum(gammaln(counts + 1), axis=1)
        max_log = np.max(log_omega)
        omega_teorico = np.exp(log_omega - max_log)
        
        # CORRECCIÓN: Ponderación Inversa por Frecuencia Empírica
        counts_rounded = np.round(counts, 4)
        _, inverse_idx, freq_empirica = np.unique(counts_rounded, axis=0, return_inverse=True, return_counts=True)
        f_emp = freq_empirica[inverse_idx]
        
        omega_corregido = omega_teorico / f_emp
        
        p_i = np.where(m_valid > 1, omega_corregido, 0.0)
        
        sum_pi = np.sum(p_i)
        if sum_pi > 0:
            p_i = p_i / sum_pi 
        else:
            p_i = np.ones(n) / n 
        
        ni_ponderado = _compute_ni_sp_vectorized(X[None, :, :], p_i[None, :], k_escala)[0]
        
        indices = np.random.choice(n, size=(S_replicas, n), replace=True, p=p_i)
        X_replicas = X[indices]
        w_flat_replicas = np.ones((S_replicas, n)) / n 
        ni_replicas = _compute_ni_sp_vectorized(X_replicas, w_flat_replicas, k_escala)
        
    else:
        indices = np.random.choice(n, size=(S_replicas, n), replace=True)
        X_replicas = X[indices]
        w_flat_replicas = np.ones((S_replicas, n)) / n
        ni_replicas = _compute_ni_sp_vectorized(X_replicas, w_flat_replicas, k_escala)
    
    mascara_validos = ~np.isnan(ni_replicas)
    ni_replicas_valid = ni_replicas[mascara_validos]
    X_replicas_valid = X_replicas[mascara_validos]
    indices_valid = indices[mascara_validos]
    
    if len(ni_replicas_valid) < 2: 
        return float(ni_muestra), float(ni_ponderado), np.nan, np.nan, ni_replicas_valid, X_replicas_valid, indices_valid
        
    return float(ni_muestra), float(ni_ponderado), float(np.percentile(ni_replicas_valid, 2.5)), float(np.percentile(ni_replicas_valid, 97.5)), ni_replicas_valid, X_replicas_valid, indices_valid

def detectar_anomalias_ni(matriz_entrada, k_escala=5, umbral_sigma=1.0):
    X = np.array(matriz_entrada, dtype=float)
    m_valid = np.sum(~np.isnan(X), axis=1)
    n_ext1 = m_valid // 2; n_ext2 = m_valid - n_ext1
    safe_m = np.where(m_valid == 0, 1, m_valid)
    mean_ext = (n_ext1 * 1 + n_ext2 * k_escala) / safe_m
    var_ext = (n_ext1 * (1 - mean_ext)**2 + n_ext2 * (k_escala - mean_ext)**2) / safe_m
    max_sigma_row = np.where(m_valid > 1, np.sqrt(var_ext), np.nan)
    sigma_row = np.nanstd(X, axis=1)
    acuerdo_local = np.where(max_sigma_row > 0, 1.0 - (sigma_row / max_sigma_row), np.nan)
    
    valid = ~np.isnan(acuerdo_local)
    if not np.any(valid): return pd.DataFrame(), np.nan, np.nan, np.nan
    mu_global = np.mean(acuerdo_local[valid])
    sigma_global = np.std(acuerdo_local[valid])
    limite = mu_global - (umbral_sigma * sigma_global)
    anomalo = (acuerdo_local < limite) & valid
    df_anomalias = pd.DataFrame({'Sujeto_ID': np.arange(1, len(X) + 1), 'Acuerdo_Local': acuerdo_local, 'Es_Anomalo': anomalo})
    return df_anomalias, mu_global, sigma_global, limite

def calcular_azar_termodinamico_ni(m_jueces, k_escala=5):
    macro_dict = _build_macrostate_dictionary(m_jueces, k_escala)
    
    acuerdos = np.array(list(macro_dict.keys()))
    w_topologico = np.array(list(macro_dict.values()))
    
    sum_w = np.sum(w_topologico)
    mu_global = np.sum(w_topologico * acuerdos) / sum_w
    var_global = np.sum(w_topologico * (acuerdos - mu_global)**2) / sum_w
    sigma_global = np.sqrt(var_global)
    
    return np.sqrt(max(0.0, mu_global * (1.0 - sigma_global)))

def calcular_percentil_universal_ni(ni_empirico, m_jueces, k_escala=5):
    macro_dict = _build_macrostate_dictionary(m_jueces, k_escala)
    
    agrupados = defaultdict(float)
    total_espacio = sum(macro_dict.values()) 
    
    for acuerdo, multiplicidad_ce in macro_dict.items():
        ni_val = np.sqrt(acuerdo)
        agrupados[round(ni_val, 6)] += multiplicidad_ce
        
    ordenados = sorted(agrupados.items())
    percentiles = []
    acumulado = 0.0
    for ni_val, mult in ordenados:
        acumulado += mult
        percentil_sup = acumulado / total_espacio
        percentiles.append({'ni': ni_val, 'p_sup': percentil_sup})
        
    ni_lower = None; p_lower = None
    ni_upper = None; p_upper = None
    
    for p in percentiles:
        if p['ni'] <= ni_empirico:
            ni_lower = p['ni']; p_lower = p['p_sup']
    for p in reversed(percentiles):
        if p['ni'] >= ni_empirico:
            ni_upper = p['ni']; p_upper = p['p_sup']
            
    if ni_lower is None: return 0.0
    if ni_upper is None: return 100.0
    if ni_lower == ni_upper: return p_lower * 100.0
    
    p_interp = p_lower + (p_upper - p_lower) * (ni_empirico - ni_lower) / (ni_upper - ni_lower)
    return p_interp * 100.0
    
# ==============================================================================
# FUNCIONES DE POBLACIÓN TEÓRICA (Para Simulaciones y Stress Testing)
# ==============================================================================

def calcular_parametros_poblacion_fraccionaria(matriz_entrada, multiplicador=1, k_escala=5):
    """
    Prepara el espacio base de la población. Si hay multiplicador, proyecta 
    el espacio de forma asintótica.
    """
    X = np.array(matriz_entrada, dtype=float)
    if multiplicador > 1:
        X = np.repeat(X, multiplicador, axis=0)
    
    df_pob = pd.DataFrame(X)
    cols_x = df_pob.columns.tolist()
    return df_pob, cols_x

def calcular_ni_poblacional_teorico(df_pob_frac, cols_x, k_escala=5):
    """
    Calcula el valor determinista del Coeficiente Natural (NI) sobre el 
    espacio poblacional completo, actuando como Ground Truth.
    """
    X = df_pob_frac[cols_x].values
    n = len(X)
    
    # Aplicamos un peso plano (uniforme) a toda la matriz poblacional
    w_plano = np.ones((1, n)) / n 
    
    # Reutilizamos el motor ultra-optimizado vectorizado
    ni_teorico = _compute_ni_sp_vectorized(X[None, :, :], w_plano, k_escala)[0]
    
    return float(ni_teorico)    