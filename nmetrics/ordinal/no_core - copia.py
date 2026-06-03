import numpy as np
import pandas as pd
import math
import itertools
from collections import Counter, defaultdict
from scipy.special import gammaln
from scipy.stats import norm

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
def _calcular_gravedad_rangos(n_sujetos, k_escala):
    if n_sujetos > 150:
        return {r: 1.0 / k_escala for r in range(1, k_escala + 1)}
        
    def stirling_2(n, k):
        if n == 0 or k == 0: return 0
        suma = sum(((-1) ** (k - i)) * math.comb(k, i) * (i ** n) for i in range(k + 1))
        return suma // math.factorial(k)
        
    P_R = {}
    for R in range(1, min(n_sujetos, k_escala) + 1):
        P_R[R] = (math.comb(k_escala, R) * math.factorial(R) * stirling_2(n_sujetos, R)) / (k_escala ** n_sujetos)
        
    P_r = {}
    for r in range(1, k_escala + 1):
        P_r[r] = sum(P_R[R] / R for R in range(r, min(n_sujetos, k_escala) + 1))
        
    return P_r

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
    
    # Grupo de Jueces con repetición (a_high) -> Coinciden q veces
    jueces_high = r * (q + 1)
    a_high = q / (m_jueces - 1)
    
    # Grupo de Jueces sin repetición extra (a_low) -> Coinciden q-1 veces
    jueces_low = (k_escala - r) * q
    a_low = (q - 1) / (m_jueces - 1)
    
    # Media y Varianza poblacional de los acuerdos de los jueces
    mu_a = (jueces_high * a_high + jueces_low * a_low) / m_jueces
    e_a2 = (jueces_high * (a_high**2) + jueces_low * (a_low**2)) / m_jueces
    
    var_a = max(0.0, e_a2 - mu_a**2)
    sigma_a = np.sqrt(var_a)
    
    # Cálculo final del NO Mínimo
    return float(np.sqrt(max(0.0, mu_a * (1.0 - sigma_a))))

def analizar_termodinamica_no(n_sujetos, m_jueces, k_escala, valor_observado=None, percentiles=[10, 50, 90, 95, 99]):
    P_r = _calcular_gravedad_rangos(n_sujetos, k_escala)
    
    prob_por_acuerdo = defaultdict(float)
    max_coincidencias = (m_jueces * (m_jueces - 1)) / 2
    
    # 🚀 ATAJO TERMODINÁMICO: Macroestados en lugar de iterar cada juez O(k^m)
    for forma in itertools.combinations_with_replacement(range(1, k_escala + 1), m_jueces):
        # 1. Probabilidad base de esta forma pura
        p_tupla_base = np.prod([P_r[r] for r in forma])
        
        if p_tupla_base > 0:
            counts = {}
            for r in forma: counts[r] = counts.get(r, 0) + 1
            
            # 2. Multiplicidad: ¿De cuántas formas se pueden ordenar estos votos?
            denom = 1.0
            for c in counts.values(): denom *= math.factorial(c)
            multiplicidad = math.factorial(m_jueces) / denom
            
            # 3. Probabilidad total del macroestado
            p_total = p_tupla_base * multiplicidad
            
            # 4. Cálculo del acuerdo
            coincidencias = sum((c * (c - 1)) / 2 for c in counts.values())
            acuerdo = round(coincidencias / max_coincidencias, 8) if max_coincidencias > 0 else 0.0
            prob_por_acuerdo[acuerdo] += p_total
            
    acuerdos_raw = np.array(list(prob_por_acuerdo.keys()))
    probs = np.array(list(prob_por_acuerdo.values()))
    probs = probs / np.sum(probs)
    
    mu_A = np.sum(acuerdos_raw * probs)
    azar_esperado = np.sqrt(mu_A)
    
    var_A = np.sum(probs * (acuerdos_raw - mu_A)**2)
    sigma_matriz = np.sqrt(max(0.0, var_A / n_sujetos))
    
    # El Suelo de Cristal es una propiedad física absoluta, independiente de sigma
    min_N = calcular_suelo_cristal_no(m_jueces, k_escala)
    
    if sigma_matriz < 1e-9:
        escalera = {f"{p}%": float(azar_esperado) for p in percentiles}
        percentil_obs = 100.0 if valor_observado is not None and valor_observado >= azar_esperado else 0.0
        return float(azar_esperado), escalera, percentil_obs, float(min_N)

    dist_matriz = norm(loc=mu_A, scale=sigma_matriz)
    
    escalera = {}
    for p in percentiles:
        val_acuerdo = dist_matriz.ppf(p / 100.0)
        escalera[f"{p}%"] = float(np.sqrt(max(0.0, val_acuerdo)))
        
    percentil_obs = 0.0
    if valor_observado is not None:
        acuerdo_obs = valor_observado ** 2
        percentil_obs = float(dist_matriz.cdf(acuerdo_obs) * 100)
        
    return float(azar_esperado), escalera, percentil_obs, float(min_N)

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