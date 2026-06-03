import numpy as np
import pandas as pd
import math
import itertools
from collections import defaultdict
from scipy.special import gammaln

# ==============================================================================
# 1. ESCÁNER DE ANOMALÍAS ORDINALES
# ==============================================================================
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
# 2. MOTOR TERMODINÁMICO GEOMÉTRICO (Azar y Percentiles Analíticos)
# ==============================================================================
def calcular_suelo_cristal_no(m_jueces, k_escala):
    """
    Teorema del Límite de Entropía Máxima para NO (Acuerdo Cruzado).
    Calcula el mínimo NO geométricamente posible asumiendo
    una distribución cíclica perfecta de los rankings de los jueces.
    """
    if m_jueces <= k_escala:
        return 0.0
        
    q = m_jueces // k_escala
    r = m_jueces % k_escala
    
    jueces_high = r * (q + 1)
    a_high = q / (m_jueces - 1)
    
    jueces_low = (k_escala - r) * q
    a_low = (q - 1) / (m_jueces - 1)
    
    mu_a = (jueces_high * a_high + jueces_low * a_low) / m_jueces
    e_a2 = (jueces_high * (a_high**2) + jueces_low * (a_low**2)) / m_jueces
    
    var_a = max(0.0, e_a2 - mu_a**2)
    sigma_a = np.sqrt(var_a)
    
    return float(np.sqrt(max(0.0, mu_a * (1.0 - sigma_a))))

def analizar_termodinamica_no(n_sujetos, m_jueces, k_escala, valor_observado=None, percentiles=[10, 50, 90, 95, 99]):
    """
    Construye la escalera topológica exacta mediante las particiones de m en <= k partes.
    El Azar se fija invariablemente como la asintota termodinámica absoluta.
    """
    # 1. AXIOMA UNIVERSAL: El Azar Teórico es INMUTABLE (Independiente de n)
    azar_esperado = np.sqrt(1.0 / k_escala)
    min_N = calcular_suelo_cristal_no(m_jueces, k_escala)
    
    prob_por_no = defaultdict(float)
    max_coincidencias = (m_jueces * (m_jueces - 1)) / 2
    p_base = (1.0 / k_escala) ** m_jueces

    # 2. Generar el hiperespacio topológico de las filas (Colapso Combinatorio)
    for forma in itertools.combinations_with_replacement(range(1, k_escala + 1), m_jueces):
        counts = {}
        for r in forma: 
            counts[r] = counts.get(r, 0) + 1
        
        # Multiplicidad termodinámica
        denom = 1.0
        for c in counts.values(): denom *= math.factorial(c)
        multiplicidad = math.factorial(m_jueces) / denom
        
        p_total = p_base * multiplicidad
        
        # Extracción del NO_local de esta clase de equivalencia (Partición)
        A_j = []
        for c in counts.values():
            if c > 0:
                a_j = (c - 1) / (m_jueces - 1) if m_jueces > 1 else 0.0
                A_j.extend([a_j] * c)
                
        mu_local = np.mean(A_j)
        sigma_local = np.std(A_j)
        no_local = np.sqrt(max(0.0, mu_local * (1.0 - sigma_local)))
        
        prob_por_no[no_local] += p_total
        
    # 3. Construir la Función de Distribución Acumulada (CDF)
    no_ordenados = np.array(sorted(prob_por_no.keys()))
    probs = np.array([prob_por_no[v] for v in no_ordenados])
    probs = probs / np.sum(probs) 
    cdf = np.cumsum(probs)
    
    # Extraer percentiles fijos para la tabla
    escalera = {}
    for p in percentiles:
        idx = np.searchsorted(cdf, p / 100.0)
        if idx >= len(no_ordenados): idx = len(no_ordenados) - 1
        escalera[f"{p}%"] = float(no_ordenados[idx])
        
    # 4. Interpolación Lineal para el Percentil Observado
    percentil_obs = 0.0
    if valor_observado is not None:
        if valor_observado <= no_ordenados[0]:
            percentil_obs = 0.0
        elif valor_observado >= no_ordenados[-1]:
            percentil_obs = 100.0
        else:
            idx = np.searchsorted(no_ordenados, valor_observado)
            no_sup = no_ordenados[idx]
            no_inf = no_ordenados[idx - 1]
            cdf_sup = cdf[idx]
            cdf_inf = cdf[idx - 1] if idx > 0 else 0.0
            
            rango_no = no_sup - no_inf
            if rango_no > 1e-9:
                frac = (valor_observado - no_inf) / rango_no
                percentil_obs = (cdf_inf + frac * (cdf_sup - cdf_inf)) * 100.0
            else:
                percentil_obs = cdf_sup * 100.0

    return float(azar_esperado), escalera, float(percentil_obs), float(min_N)

# ==============================================================================
# 3. MOTOR DE INFERENCIA UNIFICADA (Big Data Ready)
# ==============================================================================
def calcular_estadisticas_no_unificada(dict_estados, k_escala, replicas=1000):
    estados_lista = list(dict_estados.keys())
    f_t = np.array(list(dict_estados.values()), dtype=float)
    n_total = int(np.sum(f_t))
    
    try:
        X_estados = np.array([list(e) for e in estados_lista], dtype=float)
    except ValueError:
        raise ValueError("La matriz contiene datos no numéricos.")

    U, m = X_estados.shape

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

    def _no_desde_pesos(w):
        if w.ndim == 1: w = w[None, :]
        C_j = np.dot(w, match_u_j)
        P_j = np.dot(w, pos_u_j)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            A_j = np.where(P_j > 0, C_j / P_j, np.nan)
            
        mu = np.nanmean(A_j, axis=1)
        sigma = np.nanstd(A_j, axis=1) 
        
        return np.sqrt(np.maximum(mu * (1.0 - sigma), 0.0))

    w_muestra = f_t / n_total
    no_muestra = _no_desde_pesos(w_muestra)[0]

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

    log_omega = gammaln(m_valid + 1) - np.sum(gammaln(counts + 1), axis=1)
    omega_teorico = np.exp(log_omega - np.max(log_omega))

    w_pob = np.where((m_valid > 1) & (f_M_mapped > 0), f_t * (omega_teorico / f_M_mapped), 0.0)
    sum_wpob = np.sum(w_pob)
    w_pob = w_pob / sum_wpob if sum_wpob > 0 else np.ones(U) / U

    no_poblacion = _no_desde_pesos(w_pob)[0]

    safe_n = max(n_total, 2)
    
    features = np.hstack([match_u_j, pos_u_j])
    unique_features, inverse_idx = np.unique(features, axis=0, return_inverse=True)
    
    w_pob_grouped = np.zeros(len(unique_features))
    np.add.at(w_pob_grouped, inverse_idx, w_pob)
    
    match_u_j_grouped = unique_features[:, :m]
    pos_u_j_grouped = unique_features[:, m:]
    
    def _no_desde_pesos_grouped(w_grouped):
        if w_grouped.ndim == 1: w_grouped = w_grouped[None, :]
        C_j = np.dot(w_grouped, match_u_j_grouped)
        P_j = np.dot(w_grouped, pos_u_j_grouped)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            A_j = np.where(P_j > 0, C_j / P_j, np.nan)
            
        mu = np.nanmean(A_j, axis=1)
        sigma = np.nanstd(A_j, axis=1)
        return np.sqrt(np.maximum(mu * (1.0 - sigma), 0.0))

    f_boot = np.random.multinomial(safe_n, w_pob_grouped, size=replicas)
    w_boot = f_boot / safe_n
    sims = _no_desde_pesos_grouped(w_boot)

    return float(no_muestra), float(no_poblacion), float(np.percentile(sims, 2.5)), float(np.percentile(sims, 97.5))