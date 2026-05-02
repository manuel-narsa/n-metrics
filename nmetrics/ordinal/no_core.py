import numpy as np
import pandas as pd
import math
import itertools
from collections import Counter, defaultdict
from scipy.special import gammaln

# ==============================================================================
# 1. EVALUADOR BASE (Acuerdo Ordinal NO) y SIMULACIÓN PONDERADA (SP)
# ==============================================================================
def _calcular_pesos_termodinamicos_jueces(X_3d, k_escala):
    S, n, m = X_3d.shape
    counts = np.zeros((S, m, k_escala))
    X_cat = np.floor(X_3d + 0.5)
    X_cat = np.clip(X_cat, 1, k_escala)
    
    for k_val in range(1, k_escala + 1):
        counts[:, :, k_val-1] = np.sum(X_cat == k_val, axis=1)
        
    R = np.sum(counts > 0, axis=2)
    n_valid = np.sum(~np.isnan(X_3d), axis=1)
    
    # 1. Multiplicidad Termodinámica Teórica
    log_comb = gammaln(k_escala + 1) - gammaln(R + 1) - gammaln(k_escala - R + 1)
    log_multinomial = gammaln(n_valid + 1) - np.sum(gammaln(counts + 1), axis=2)
    
    log_omega = log_comb + log_multinomial
    max_log = np.max(log_omega, axis=1, keepdims=True)
    omega_teorico = np.exp(log_omega - max_log)
    
    # 2. CORRECCIÓN DE DOBLE PONDERACIÓN (Frecuencia Empírica de Columnas)
    f_emp = np.ones((S, m))
    counts_rounded = np.round(counts, 4)
    for s in range(S):
        # Buscamos jueces (columnas) con la misma firma exacta en esta iteración
        _, inverse_idx, freq = np.unique(counts_rounded[s], axis=0, return_inverse=True, return_counts=True)
        f_emp[s] = freq[inverse_idx]
        
    # Aplicamos la Ponderación Inversa
    omega_corregido = omega_teorico / f_emp
    
    # 3. Normalización final
    sum_omega = np.sum(omega_corregido, axis=1, keepdims=True)
    pesos = np.where(sum_omega > 0, omega_corregido / sum_omega, 1.0 / m)
    return pesos

def _compute_no_sp_vectorized(X_3d, w_pesos_jueces, k_escala):
    S, n, m = X_3d.shape
    X_cat = np.floor(X_3d + 0.5)
    X_cat = np.clip(X_cat, 1, k_escala)
    
    X_sm = X_cat.transpose(0, 2, 1).reshape(S * m, n)
    sort_idx = np.argsort(X_sm, axis=1)
    row_idx = np.arange(S * m)[:, None]
    sorted_X = X_sm[row_idx, sort_idx]
    
    diffs = np.diff(sorted_X, axis=1)
    diffs[np.isnan(diffs)] = 0.0
    step = np.concatenate([np.zeros((S * m, 1)), (diffs != 0).astype(int)], axis=1)
    ranks = (np.cumsum(step, axis=1) + 1)
    
    inv_sort_idx = np.argsort(sort_idx, axis=1)
    restored_ranks = ranks[row_idx, inv_sort_idx]
    R_sm = restored_ranks.astype(float)
    R_sm[np.isnan(X_sm)] = np.nan 
    
    R_3d_jueces = R_sm.reshape(S, m, n) 
    
    R_A = R_3d_jueces[:, :, None, :] 
    R_B = R_3d_jueces[:, None, :, :] 
    
    eye_mask = ~np.eye(m, dtype=bool)[None, :, :, None] 
    
    valid_mask = ~np.isnan(R_A) & ~np.isnan(R_B) & eye_mask
    matches = (R_A == R_B) & valid_mask
    
    coincidencias_juez = np.sum(matches, axis=(2, 3))       
    emparejamientos_juez = np.sum(valid_mask, axis=(2, 3)) 
    
    acuerdos_juez = np.divide(coincidencias_juez, emparejamientos_juez, 
                          out=np.full_like(coincidencias_juez, np.nan, dtype=float), 
                          where=emparejamientos_juez > 0)
    
    valid_rows = ~np.isnan(acuerdos_juez)
    pesos_validos = np.where(valid_rows, w_pesos_jueces, 0.0)
    
    sum_pesos = np.sum(pesos_validos, axis=1, keepdims=True)
    pesos_norm = np.where(sum_pesos > 0, pesos_validos / sum_pesos, 0.0)
    
    mu_global = np.sum(acuerdos_juez * pesos_norm, axis=1)
    var_global = np.sum(pesos_norm * (acuerdos_juez - mu_global[:, None])**2, axis=1)
    sigma_global = np.sqrt(var_global)
    
    return np.sqrt(np.maximum(mu_global * (1 - sigma_global), 0.0))

def calcular_estadisticas_no(matriz_entrada, S_replicas, k_escala=5, metodo_ic='SP'):
    X = np.array(matriz_entrada, dtype=float)
    n, m = X.shape
    
    w_plano_jueces = np.ones((1, m)) / m
    no_muestra = _compute_no_sp_vectorized(X[None, :, :], w_plano_jueces, k_escala)[0]
    
    w_termo_jueces = _calcular_pesos_termodinamicos_jueces(X[None, :, :], k_escala)
    no_ponderado = _compute_no_sp_vectorized(X[None, :, :], w_termo_jueces, k_escala)[0]
    
    indices = np.random.choice(n, size=(S_replicas, n), replace=True)
    X_replicas = X[indices]
    
    if metodo_ic == 'SP':
        w_termo_replicas = _calcular_pesos_termodinamicos_jueces(X_replicas, k_escala)
        no_replicas = _compute_no_sp_vectorized(X_replicas, w_termo_replicas, k_escala)
    else:
        w_flat_replicas = np.ones((S_replicas, m)) / m
        no_replicas = _compute_no_sp_vectorized(X_replicas, w_flat_replicas, k_escala)
        
    mascara_validos = ~np.isnan(no_replicas)
    no_replicas_valid = no_replicas[mascara_validos]
    X_replicas_valid = X_replicas[mascara_validos]
    indices_valid = indices[mascara_validos]
    
    if len(no_replicas_valid) < 2: 
        return float(no_muestra), float(no_ponderado), np.nan, np.nan, no_replicas_valid, X_replicas_valid, indices_valid
    return float(no_muestra), float(no_ponderado), float(np.percentile(no_replicas_valid, 2.5)), float(np.percentile(no_replicas_valid, 97.5)), no_replicas_valid, X_replicas_valid, indices_valid

def detectar_anomalias_no(matriz_entrada, k_escala=5, umbral_sigma=1.0):
    X = np.array(matriz_entrada, dtype=float)
    X_cat = np.floor(X + 0.5); X_cat = np.clip(X_cat, 1, k_escala)
    n, m = X.shape
    
    X_sm = X_cat.T 
    sort_idx = np.argsort(X_sm, axis=1)
    row_idx = np.arange(m)[:, None]
    sorted_X = X_sm[row_idx, sort_idx]
    
    diffs = np.diff(sorted_X, axis=1); diffs[np.isnan(diffs)] = 0.0
    step = np.concatenate([np.zeros((m, 1)), (diffs != 0).astype(int)], axis=1)
    ranks = (np.cumsum(step, axis=1) + 1)
    
    inv_sort_idx = np.argsort(sort_idx, axis=1)
    R_mn = ranks[row_idx, inv_sort_idx].astype(float) 
    R_mn[np.isnan(X.T)] = np.nan
    
    R_A = R_mn[:, None, :] 
    R_B = R_mn[None, :, :] 
    
    valid_A = ~np.isnan(R_A)
    valid_B = ~np.isnan(R_B)
    
    eye_mask = ~np.eye(m, dtype=bool)[:, :, None] 
    
    valid_pairs = valid_A & valid_B & eye_mask
    matches = (R_A == R_B) & valid_pairs
    
    coincidencias_juez = np.sum(matches, axis=(1, 2))
    emparejamientos_juez = np.sum(valid_pairs, axis=(1, 2))
    
    acuerdo_local = np.divide(coincidencias_juez, emparejamientos_juez,
                              out=np.full_like(coincidencias_juez, np.nan, dtype=float),
                              where=emparejamientos_juez > 0)
    
    valid = ~np.isnan(acuerdo_local)
    if not np.any(valid): 
        return pd.DataFrame(), np.nan, np.nan, np.nan
    
    mu_global = np.mean(acuerdo_local[valid])
    sigma_global = np.std(acuerdo_local[valid])
    limite = mu_global - (umbral_sigma * sigma_global)
    anomalo = (acuerdo_local < limite) & valid
    
    df_anomalias = pd.DataFrame({'Juez_ID': np.arange(1, m + 1), 'Acuerdo_Local': acuerdo_local, 'Es_Anomalo': anomalo})
    return df_anomalias, mu_global, sigma_global, limite

# ==============================================================================
# 2. MOTOR TERMODINÁMICO EXACTO (Azar y Percentiles Combinatorios)
# ==============================================================================
_MACRO_DICT_CACHE = {}  # ¡La clave del rendimiento: Memoria RAM!

def _probabilidad_rango_denso(r, k, n):
    suma_total = 0
    for v in range(r, k + 1):
        comb_v = math.comb(v - 1, r - 1)
        suma_inc_exc = 0
        for j in range(r):
            termino = ((-1)**j) * math.comb(r - 1, j) * ((r + k - v - j)**(n - 1))
            suma_inc_exc += termino
        suma_total += comb_v * suma_inc_exc
    return float(suma_total) / float(k**n)

def _build_macrostate_dictionary_no_exacto(n_sujetos, m_jueces, k_escala):
    key = (n_sujetos, k_escala)
    if key in _MACRO_DICT_CACHE:
        return _MACRO_DICT_CACHE[key]
        
    F = np.array([_probabilidad_rango_denso(r, k_escala, n_sujetos) for r in range(1, k_escala + 1)])
    macro_dict = defaultdict(float)
    
    log_k_n = n_sujetos * np.log(k_escala)
    log_n_fact = gammaln(n_sujetos + 1)
    
    for num_rangos in range(1, k_escala + 1):
        comb_k = math.comb(k_escala, num_rangos)
        log_comb_k = np.log(comb_k)
        
        if num_rangos == 1:
            comps = np.array([[n_sujetos]])
            esperanzas = np.dot(comps, F[:num_rangos])
            acuerdos = np.round(esperanzas / n_sujetos, 8)
            
            log_denom = np.sum(gammaln(comps + 1), axis=1)
            log_probs = log_comb_k + log_n_fact - log_denom - log_k_n
            probs = np.exp(log_probs)
            
            for ac, p in zip(acuerdos, probs):
                macro_dict[ac] += p
        else:
            iterator = itertools.combinations(range(1, n_sujetos), num_rangos - 1)
            # Procesamos en bloques para que la memoria RAM no explote con matrices masivas
            CHUNK_SIZE = 1_000_000
            while True:
                chunk = list(itertools.islice(iterator, CHUNK_SIZE))
                if not chunk:
                    break
                    
                indices = np.array(chunk)
                comps = np.zeros((len(indices), num_rangos), dtype=int)
                comps[:, 0] = indices[:, 0]
                comps[:, 1:-1] = np.diff(indices, axis=1)
                comps[:, -1] = n_sujetos - indices[:, -1]
                
                esperanzas = np.dot(comps, F[:num_rangos])
                acuerdos = np.round(esperanzas / n_sujetos, 8)
                
                # Fórmula topológica vectorizada:
                log_denom = np.sum(gammaln(comps + 1), axis=1)
                log_probs = log_comb_k + log_n_fact - log_denom - log_k_n
                probs = np.exp(log_probs)
                
                unique_acuerdos, inverse_indices = np.unique(acuerdos, return_inverse=True)
                sum_probs = np.bincount(inverse_indices, weights=probs)
                
                for ac, p in zip(unique_acuerdos, sum_probs):
                    macro_dict[ac] += p
                    
    _MACRO_DICT_CACHE[key] = macro_dict
    return macro_dict

def calcular_azar_termodinamico_no_analitico_exacto(n_sujetos, m_jueces, k_escala=5):
    macro_dict = _build_macrostate_dictionary_no_exacto(n_sujetos, m_jueces, k_escala)
    
    acuerdos = np.array(list(macro_dict.keys()))
    probabilidades = np.array(list(macro_dict.values()))
    
    mu_global = np.sum(acuerdos * probabilidades)
    var_global = np.sum(probabilidades * (acuerdos - mu_global)**2)
    sigma_global = np.sqrt(var_global)
    
    no_azar_esperado = np.sqrt(max(0.0, mu_global * (1.0 - sigma_global)))
    
    return float(no_azar_esperado), 0.0, 0.0

def calcular_percentil_universal_no_exacto(no_empirico, n_sujetos, m_jueces, k_escala=5):
    macro_dict = _build_macrostate_dictionary_no_exacto(n_sujetos, m_jueces, k_escala)
    
    acuerdos = np.array(list(macro_dict.keys()))
    probabilidades = np.array(list(macro_dict.values()))
    
    # 1. Calculamos la penalización topológica global
    mu_ref = np.sum(acuerdos * probabilidades)
    sigma_ref = np.sqrt(np.sum(probabilidades * (acuerdos - mu_ref)**2))
    
    # 2. Construimos la "Escalera de Probabilidad" agrupando nodos discretos
    agrupados = defaultdict(float)
    total_espacio = np.sum(probabilidades)
    
    for acuerdo, prob in macro_dict.items():
        no_val = np.sqrt(max(0.0, acuerdo * (1.0 - sigma_ref)))
        agrupados[round(no_val, 6)] += prob
        
    ordenados = sorted(agrupados.items())
    percentiles = []
    acumulado = 0.0
    
    for no_val, prob in ordenados:
        acumulado += prob
        percentil_sup = acumulado / total_espacio
        percentiles.append({'no': no_val, 'p_sup': percentil_sup})
        
    # 3. Interpolación Lineal (Universal Percentile)
    # ¡LA CLAVE! Redondeamos el empírico a la misma resolución para evitar saltos de coma flotante
    no_emp = round(float(no_empirico), 6)
    
    no_lower = None; p_lower = None
    no_upper = None; p_upper = None
    
    for p in percentiles:
        if p['no'] <= no_emp:
            no_lower = p['no']; p_lower = p['p_sup']
        for p in reversed(percentiles):
            if p['no'] >= no_emp:
                no_upper = p['no']; p_upper = p['p_sup']
                
    # Array de compatibilidad para el retorno en la interfaz antigua
    array_no_azar = np.sqrt(np.maximum(acuerdos * (1 - sigma_ref), 0.0))
            
    if no_lower is None: return 0.0, array_no_azar
    if no_upper is None: return 100.0, array_no_azar
    
    if no_lower == no_upper:
        # MANEJO DE SINGULARIDAD TOPOLÓGICA
        # Si el hiperespacio colapsa en un único punto (Varianza 0), 
        # su centro de masa absoluto por azar es el percentil 50.
        if len(percentiles) == 1:
            return 50.0, array_no_azar
        return p_lower * 100.0, array_no_azar
    
    # Ejecutamos la interpolación lineal estricta entre nodos adyacentes
    p_interp = p_lower + (p_upper - p_lower) * (no_emp - no_lower) / (no_upper - no_lower)
    return p_interp * 100.0, array_no_azar