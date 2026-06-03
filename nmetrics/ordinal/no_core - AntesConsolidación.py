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
_MACRO_DICT_CACHE = {}

def _probabilidad_rango_denso(r, k, n):
    suma_total = 0
    for v in range(r, k + 1):
        comb_v = math.comb(v - 1, r - 1)
        suma_inc_exc = 0
        for j in range(r):
            termino = ((-1)**j) * math.comb(r - 1, j) * ((r + k - v - j)**(n - 1))
            suma_inc_exc += termino
        suma_total += comb_v * suma_inc_exc
        
    if suma_total == 0: return 0.0
    
    # CORRECCIÓN BIG DATA: Espacio Logarítmico para evitar OverflowError con float(k**n)
    log_prob = math.log(suma_total) - (n * math.log(k))
    try:
        return math.exp(log_prob)
    except OverflowError:
        return 0.0

def _build_macrostate_dictionary_no_exacto(n_sujetos, m_jueces, k_escala):
    key = (n_sujetos, k_escala)
    if key in _MACRO_DICT_CACHE: return _MACRO_DICT_CACHE[key]
        
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
            for ac, p in zip(acuerdos, probs): macro_dict[ac] += p
        else:
            iterator = itertools.combinations(range(1, n_sujetos), num_rangos - 1)
            CHUNK_SIZE = 1_000_000
            while True:
                chunk = list(itertools.islice(iterator, CHUNK_SIZE))
                if not chunk: break
                indices = np.array(chunk)
                comps = np.zeros((len(indices), num_rangos), dtype=int)
                comps[:, 0] = indices[:, 0]
                comps[:, 1:-1] = np.diff(indices, axis=1)
                comps[:, -1] = n_sujetos - indices[:, -1]
                
                esperanzas = np.dot(comps, F[:num_rangos])
                acuerdos = np.round(esperanzas / n_sujetos, 8)
                
                log_denom = np.sum(gammaln(comps + 1), axis=1)
                log_probs = log_comb_k + log_n_fact - log_denom - log_k_n
                probs = np.exp(log_probs)
                
                unique_acuerdos, inverse_indices = np.unique(acuerdos, return_inverse=True)
                sum_probs = np.bincount(inverse_indices, weights=probs)
                for ac, p in zip(unique_acuerdos, sum_probs): macro_dict[ac] += p
                    
    _MACRO_DICT_CACHE[key] = macro_dict
    return macro_dict

def calcular_azar_termodinamico_no_analitico_exacto(n_sujetos, m_jueces, k_escala=5):
    """
    El azar en la topología Ordinal (NO) es una constante exacta del sistema 
    independiente de N. No requiere estimación asintótica.
    """
    no_azar_esperado = np.sqrt(1.0 / k_escala)
    return float(no_azar_esperado), 0.0, 0.0

def calcular_percentil_universal_no_exacto(no_empirico, n_sujetos, m_jueces, k_escala=5):
    """
    Calcula el percentil. Para N > 150 aplica el colapso de la varianza (Big Data),
    para N <= 150 traza la distribución exacta del hiperespacio.
    """
    azar_exacto = np.sqrt(1.0 / k_escala)
    
    # PROTECCIÓN BIG DATA: Para N masivos, la varianza del hiperespacio colapsa a 0.
    if n_sujetos > 150:
        perc = 100.0 if no_empirico >= azar_exacto else 0.0
        return perc, np.array([azar_exacto])

    # Para matrices pequeñas, calculamos la distribución de probabilidad exacta
    macro_dict = _build_macrostate_dictionary_no_exacto(n_sujetos, m_jueces, k_escala)
    acuerdos = np.array(list(macro_dict.keys()))
    probabilidades = np.array(list(macro_dict.values()))
    
    mu_ref = np.sum(acuerdos * probabilidades)
    sigma_ref = np.sqrt(np.sum(probabilidades * (acuerdos - mu_ref)**2))
    
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
        
    no_emp = round(float(no_empirico), 6)
    no_lower = None; p_lower = None
    no_upper = None; p_upper = None
    
    for p in percentiles:
        if p['no'] <= no_emp:
            no_lower = p['no']; p_lower = p['p_sup']
        for p in reversed(percentiles):
            if p['no'] >= no_emp:
                no_upper = p['no']; p_upper = p['p_sup']
                
    array_no_azar = np.sqrt(np.maximum(acuerdos * (1 - sigma_ref), 0.0))
            
    if no_lower is None: return 0.0, array_no_azar
    if no_upper is None: return 100.0, array_no_azar
    if no_lower == no_upper:
        if len(percentiles) == 1: return 50.0, array_no_azar
        return p_lower * 100.0, array_no_azar
    
    p_interp = p_lower + (p_upper - p_lower) * (no_emp - no_lower) / (no_upper - no_lower)
    return p_interp * 100.0, array_no_azar

# ==============================================================================
# 3. MOTOR DE CONDENSACIÓN PARA BIG DATA (El algoritmo de los 6 pasos)
# ==============================================================================
def calcular_no_masivo(matriz, k_escala):
    """Estrategia de Condensación para Ordinal (NO) - Ranking por Juez"""
    # 1. Convertir a ranking denso por columna
    m_rank = np.zeros_like(matriz, dtype=float)
    m_cols = matriz.shape[1]
    
    for j in range(m_cols):
        col = matriz[:, j]
        mask = ~np.isnan(col)
        valid = col[mask]
        if len(valid) > 0:
            uniq = np.sort(np.unique(valid))
            mapping = {val: idx+1 for idx, val in enumerate(uniq)}
            m_rank[mask, j] = [mapping[v] for v in valid]
        m_rank[~mask, j] = np.nan

    # 2. Aislar filas únicas y multiplicidad
    m_filled = np.nan_to_num(m_rank, nan=-999)
    unique_rows, counts = np.unique(m_filled, axis=0, return_counts=True)

    # 3 & 4. Contar coincidencias cruzadas por cada juez (Sin contarse a sí mismo)
    acuerdos_jueces = []
    for j in range(m_cols):
        coincidencias = 0
        posibles = 0
        for row, count in zip(unique_rows, counts):
            val_j = row[j]
            if val_j == -999: continue
            
            otros = np.delete(row, j)
            validos_otros = otros[otros != -999]
            
            coincidencias += np.sum(validos_otros == val_j) * count
            posibles += len(validos_otros) * count
            
        if posibles > 0:
            acuerdos_jueces.append(coincidencias / posibles)
        
    if not acuerdos_jueces:
        return 0.0
        
    # 5 & 6. Promedio, Desviación poblacional y cálculo final de NO
    acuerdos_jueces = np.array(acuerdos_jueces)
    mu = np.mean(acuerdos_jueces)
    sigma_pob = np.std(acuerdos_jueces) 
    
    return float(np.sqrt(max(0.0, mu * (1.0 - sigma_pob))))
    
def calcular_estadisticas_no_unificada(dict_estados, k_escala, replicas=1000):
    """
    Motor universal condensado para Escala Ordinal (NO).
    Aplica el ranking denso por juez y el cálculo de consenso cruzado global,
    acelerado mediante álgebra matricial y Bootstrap Multinomial O(1).
    """
    estados_lista = list(dict_estados.keys())
    f_t = np.array(list(dict_estados.values()), dtype=float)
    n_total = int(np.sum(f_t))
    
    try:
        X_estados = np.array([list(e) for e in estados_lista], dtype=float)
    except ValueError:
        raise ValueError("La matriz contiene datos no numéricos.")

    U, m = X_estados.shape

    # ---------------------------------------------------------
    # PASOS 1, 2 y 3: RANKING DENSO Y MATRIZ DE COINCIDENCIAS
    # ---------------------------------------------------------
    R_estados = np.zeros_like(X_estados, dtype=float)
    for j in range(m):
        col = X_estados[:, j]
        mask = ~np.isnan(col)
        valid = col[mask]
        if len(valid) > 0:
            uniq = np.sort(np.unique(valid))
            mapping = {val: idx+1 for idx, val in enumerate(uniq)}
            R_estados[mask, j] = [mapping[v] for v in valid]
        R_estados[~mask, j] = np.nan

    # Pre-cálculo de emparejamientos por tupla y juez
    match_u_j = np.zeros((U, m), dtype=float)
    pos_u_j = np.zeros((U, m), dtype=float)

    for j in range(m):
        val_j = R_estados[:, j]
        valid_j = ~np.isnan(val_j)
        for k in range(m):
            if j == k: continue
            val_k = R_estados[:, k]
            valid_k = ~np.isnan(val_k)
            
            both_valid = valid_j & valid_k
            match_u_j[:, j] += (both_valid & (val_j == val_k)).astype(float)
            pos_u_j[:, j] += both_valid.astype(float)

    # Función Lambda (Pasos 4, 5 y 6)
    def _no_desde_pesos(w):
        if w.ndim == 1: w = w[None, :]
        # Matrices cruzadas con los pesos dinámicos
        C_j = np.dot(w, match_u_j)
        P_j = np.dot(w, pos_u_j)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            A_j = np.where(P_j > 0, C_j / P_j, np.nan)
            
        mu = np.nanmean(A_j, axis=1)
        sigma = np.nanstd(A_j, axis=1) # ddof=0 poblacional por defecto
        
        return np.sqrt(np.maximum(mu * (1.0 - sigma), 0.0))

    # ---------------------------------------------------------
    # 1. NO MUESTRA
    # ---------------------------------------------------------
    w_muestra = f_t / n_total
    no_muestra = _no_desde_pesos(w_muestra)[0]

    # ---------------------------------------------------------
    # 2. NO POBLACIÓN (Corrección Termodinámica Exacta)
    # ---------------------------------------------------------
    m_valid = np.sum(~np.isnan(X_estados), axis=1)
    counts = np.zeros((U, k_escala))
    X_clip = np.floor(X_estados + 0.5)
    X_clip = np.clip(X_clip, 1, k_escala)
    
    for k_val in range(1, k_escala + 1):
        counts[:, k_val-1] = np.sum(X_clip == k_val, axis=1)

    counts_rounded = np.round(counts, 4)
    unique_counts, inverse_idx = np.unique(counts_rounded, axis=0, return_inverse=True)

    f_M = np.zeros(len(unique_counts))
    np.add.at(f_M, inverse_idx, f_t)
    f_M_mapped = f_M[inverse_idx]

    # MULTIPLICIDAD EXACTA: Eliminado log_comb. 
    # Solo Función Multinomial para coincidir con la firma posicional de la tupla.
    log_omega = gammaln(m_valid + 1) - np.sum(gammaln(counts + 1), axis=1)
    omega_teorico = np.exp(log_omega - np.max(log_omega))

    w_pob = np.where((m_valid > 1) & (f_M_mapped > 0), f_t * (omega_teorico / f_M_mapped), 0.0)
    sum_wpob = np.sum(w_pob)
    w_pob = w_pob / sum_wpob if sum_wpob > 0 else np.ones(U) / U

    no_poblacion = _no_desde_pesos(w_pob)[0]

    # ---------------------------------------------------------
    # 3. IC BOOTSTRAP (Multinomial O(1))
    # ---------------------------------------------------------
    safe_n = max(n_total, 2)
    
    # 🚀 SOLUCIÓN BIG DATA: Agrupamos por la huella exacta de coincidencias cruzadas
    # Comprime los estados a unas pocas firmas topológicas
    features = np.hstack([match_u_j, pos_u_j])
    unique_features, inverse_idx = np.unique(features, axis=0, return_inverse=True)
    
    w_pob_grouped = np.zeros(len(unique_features))
    np.add.at(w_pob_grouped, inverse_idx, w_pob)
    
    # Restauramos las matrices de match y posibles pero en versión miniatura
    match_u_j_grouped = unique_features[:, :m]
    pos_u_j_grouped = unique_features[:, m:]
    
    # Lambda adaptada al espacio cruzado comprimido
    def _no_desde_pesos_grouped(w_grouped):
        if w_grouped.ndim == 1: w_grouped = w_grouped[None, :]
        C_j = np.dot(w_grouped, match_u_j_grouped)
        P_j = np.dot(w_grouped, pos_u_j_grouped)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            A_j = np.where(P_j > 0, C_j / P_j, np.nan)
            
        mu = np.nanmean(A_j, axis=1)
        sigma = np.nanstd(A_j, axis=1)
        return np.sqrt(np.maximum(mu * (1.0 - sigma), 0.0))

    # Simulación hiper-optimizada
    f_boot = np.random.multinomial(safe_n, w_pob_grouped, size=replicas)
    w_boot = f_boot / safe_n
    
    sims = _no_desde_pesos_grouped(w_boot)

    return float(no_muestra), float(no_poblacion), float(np.percentile(sims, 2.5)), float(np.percentile(sims, 97.5))