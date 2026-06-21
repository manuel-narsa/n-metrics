import numpy as np
import pandas as pd
import math
import itertools
from collections import Counter
from scipy.special import gammaln

# ==============================================================================
# 1. ESCÁNER DE ANOMALÍAS ORDINALES (Entrelazamiento Vertical)
# ==============================================================================
def detectar_anomalias_no(matriz_entrada, k_escala=5, umbral_sigma=1.0):
    """
    Detecta jueces (columnas) que introducen entropía vertical anómala
    midiendo la desviación de sus acuerdos cruzados mediante Ranking Denso.
    """
    X = np.array(matriz_entrada, dtype=float)
    X_cat = np.floor(X + 0.5); X_cat = np.clip(X_cat, 1, k_escala)
    n, m = X.shape
    
    # Conversión topológica a rangos densos
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
# 2. MOTOR TERMODINÁMICO (Espacio de Masa Discreta Exacta Compactada)
# ==============================================================================
def calcular_suelo_cristal_no(m_jueces, k_escala):
    """Calcula el mínimo NO geométricamente posible (Suelo de Cristal)."""
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

# ==============================================================================
# REEMPLAZO EN no_core.py: OPTIMIZACIÓN TOPOLÓGICA (CLASES DE EQUIVALENCIA)
# ==============================================================================
def analizar_termodinamica_no(n_sujetos, m_jueces, k_escala, valor_observado=None):
    """
    Sustituye la campana de Gauss por el colapso exacto del hiperespacio en Macroestados.
    [OPTIMIZADO MEDIANTE PARTICIONES ACOTADAS EN O(1) ESPACIAL]
    """
    import math
    import numpy as np
    from collections import defaultdict
    
    azar_esperado = float(np.sqrt(1.0 / k_escala))
    min_N = calcular_suelo_cristal_no(m_jueces, k_escala)
    
    prob_por_no = defaultdict(float)
    p_base = (1.0 / k_escala) ** m_jueces
    m_pairs = m_jueces * (m_jueces - 1) / 2.0
    
    fact = [math.factorial(i) for i in range(max(m_jueces, k_escala) + 1)]
    
    # 1. Generador ultra-rápido LIMITADO a k_escala
    def _get_bounded_partitions(n, max_len, max_val=None):
        if max_val is None: max_val = n
        if n == 0:
            yield ()
            return
        if max_len == 0:
            return
        for i in range(min(n, max_val), 0, -1):
            for p in _get_bounded_partitions(n - i, max_len - 1, i):
                yield (i,) + p

    particiones = list(_get_bounded_partitions(m_jueces, k_escala))
    
    # 2. Bucle Topológico: iteramos solo sobre las particiones válidas
    for p in particiones:
        v = len(p)
        
        # A) Coincidencias puras del macroestado (C_p)
        c_p = sum((c * (c - 1)) / 2.0 for c in p)
        
        mu_macro = c_p / m_pairs
        no_macro = np.sqrt(mu_macro) 
        
        # B) Multiplicidad base
        denom_jueces = 1.0
        for c in p:
            denom_jueces *= fact[c]
        multiplicidad_base = fact[m_jueces] / denom_jueces
        
        # C) Peso combinatorio de la escala
        counts_of_sizes = {}
        for c in p:
            counts_of_sizes[c] = counts_of_sizes.get(c, 0) + 1
            
        peso_escala = 1.0
        for i in range(v):
            peso_escala *= (k_escala - i)
            
        for size, freq in counts_of_sizes.items():
            peso_escala /= fact[freq]
            
        multiplicidad_total = multiplicidad_base * peso_escala
        p_total = p_base * multiplicidad_total
        
        prob_por_no[no_macro] += p_total

    # Función de Distribución Acumulada Discreta (CDF Física)
    no_ordenados = np.array(sorted(prob_por_no.keys()))
    probs = np.array([prob_por_no[v] for v in no_ordenados])
    probs = probs / np.sum(probs) 
    cdf = np.cumsum(probs)
    
    def _interpolar_masa(valor):
        if valor <= no_ordenados[0]: return cdf[0] * 100.0 if valor == no_ordenados[0] else 0.0
        if valor >= no_ordenados[-1]: return 100.0
        idx = np.searchsorted(no_ordenados, valor)
        no_sup = no_ordenados[idx]
        no_inf = no_ordenados[idx - 1]
        cdf_sup = cdf[idx]
        cdf_inf = cdf[idx - 1] if idx > 0 else 0.0
        
        rango_no = no_sup - no_inf
        if rango_no > 1e-9:
            frac = (valor - no_inf) / rango_no
            return (cdf_inf + frac * (cdf_sup - cdf_inf)) * 100.0
        return cdf_sup * 100.0
        
    percentil_azar_exacto = _interpolar_masa(azar_esperado)
    percentil_obs = _interpolar_masa(valor_observado) if valor_observado is not None else 0.0
    
    info_estructura = {"Percentil_Azar_Real": percentil_azar_exacto, "Suelo": min_N}
    
    return float(azar_esperado), info_estructura, float(percentil_obs), float(min_N)

# ==============================================================================
# 3. MOTOR DE INFERENCIA UNIFICADA (Producto Punto - Big Data Ready)
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

    # Conversión Topológica a Rangos Densos
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

    # Extracción de Coincidencias Cruzadas
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

    # Motor base de evaluación (Permite ponderar filas y columnas)
    def _no_desde_pesos(w_rows, w_cols=None):
        if w_rows.ndim == 1: w_rows = w_rows[None, :]
        C_j = np.dot(w_rows, match_u_j)
        P_j = np.dot(w_rows, pos_u_j)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            A_j = np.where(P_j > 0, C_j / P_j, np.nan)
            
        if w_cols is None:
            mu = np.nanmean(A_j, axis=1)
            sigma = np.nanstd(A_j, axis=1)
        else:
            if w_cols.ndim == 1: w_cols = w_cols[None, :]
            valid = ~np.isnan(A_j)
            w_valid = np.where(valid, w_cols, 0.0)
            sum_w = np.sum(w_valid, axis=1, keepdims=True)
            w_norm = np.where(sum_w > 0, w_valid / sum_w, 0.0)

            # --- CORRECCIÓN ANTI-NaNs ---
            # Transformamos los NaNs en 0.0 para que la suma matemática no se infecte
            A_j_safe = np.nan_to_num(A_j, nan=0.0)
            
            mu = np.sum(A_j_safe * w_norm, axis=1)
            var = np.sum(w_norm * (A_j_safe - mu[:, None])**2, axis=1)
            sigma = np.sqrt(var)
            
        return np.sqrt(np.maximum(mu * (1.0 - sigma), 0.0))

    # Muestra Real (Frecuencias empíricas de los sujetos, jueces con peso plano)
    no_muestra = _no_desde_pesos(f_t)[0]

    # ==============================================================================
    # Re-ponderación Bayesiana (Espacio de Configuraciones de las COLUMNAS/JUECES)
    # ==============================================================================
    counts_j = np.zeros((m, k_escala))
    X_clip = np.floor(X_estados + 0.5)
    X_clip = np.clip(X_clip, 1, k_escala)
    
    # Reconstruimos la distribución categórica que usó cada juez en la matriz real
    for j in range(m):
        valid_mask = ~np.isnan(X_clip[:, j])
        for k_val in range(1, k_escala + 1):
            counts_j[j, k_val-1] = np.sum(f_t[valid_mask & (X_clip[:, j] == k_val)])

    R_j = np.sum(counts_j > 0, axis=1)
    n_valid_j = np.zeros(m)
    for j in range(m):
        n_valid_j[j] = np.sum(f_t[~np.isnan(X_clip[:, j])])

    # 1. Multiplicidad Termodinámica Teórica por Juez (Binomial + Multinomial)
    log_comb_j = gammaln(k_escala + 1) - gammaln(R_j + 1) - gammaln(k_escala - R_j + 1)
    log_multi_j = gammaln(n_valid_j + 1) - np.sum(gammaln(counts_j + 1), axis=1)
    
    log_omega_j = log_comb_j + log_multi_j
    omega_teorico_j = np.exp(log_omega_j - np.max(log_omega_j))

    # 2. Corrección de frecuencias empíricas de las firmas de jueces
    counts_rounded_j = np.round(counts_j, 4)
    _, inverse_idx_j, freq_j = np.unique(counts_rounded_j, axis=0, return_inverse=True, return_counts=True)
    f_emp_j = freq_j[inverse_idx_j]

    # 3. Pesos poblacionales para las columnas (Jueces)
    omega_corregido_j = omega_teorico_j / f_emp_j
    sum_omega_j = np.sum(omega_corregido_j)
    w_jueces_pob = omega_corregido_j / sum_omega_j if sum_omega_j > 0 else np.ones(m) / m

    # Población Asintótica Estimada (Sujetos empíricos + Jueces ponderados topológicamente)
    no_poblacion = _no_desde_pesos(f_t, w_cols=w_jueces_pob)[0]

    # ==============================================================================
    # Bootstrap Combinatorio (Vectorizado)
    # ==============================================================================
    safe_n = max(n_total, 2)
    
    # 1. Remuestreamos las filas empíricas
    f_boot = np.random.multinomial(safe_n, f_t / n_total, size=replicas) # Forma: (replicas, U)
    
    # 2. Vectorizamos la extracción de conteos de los jueces para todas las réplicas
    counts_boot = np.zeros((replicas, m, k_escala))
    for j in range(m):
        valid_mask = ~np.isnan(X_clip[:, j])
        for k_val in range(1, k_escala + 1):
            mask = valid_mask & (X_clip[:, j] == k_val)
            counts_boot[:, j, k_val-1] = np.sum(f_boot[:, mask], axis=1)
            
    R_boot = np.sum(counts_boot > 0, axis=2) # Forma: (replicas, m)
    n_valid_boot = np.zeros((replicas, m))
    for j in range(m):
        n_valid_boot[:, j] = np.sum(f_boot[:, ~np.isnan(X_clip[:, j])], axis=1)

    # 3. Calculamos Omega para todas las réplicas simultáneamente
    log_comb_boot = gammaln(k_escala + 1) - gammaln(R_boot + 1) - gammaln(k_escala - R_boot + 1)
    log_multi_boot = gammaln(n_valid_boot + 1) - np.sum(gammaln(counts_boot + 1), axis=2)
    
    log_omega_boot = log_comb_boot + log_multi_boot
    max_log_boot = np.max(log_omega_boot, axis=1, keepdims=True)
    omega_teorico_boot = np.exp(log_omega_boot - max_log_boot)

    w_jueces_boot = np.zeros((replicas, m))
    for r in range(replicas):
        counts_r = np.round(counts_boot[r], 4)
        _, inv_idx, frq = np.unique(counts_r, axis=0, return_inverse=True, return_counts=True)
        w_jueces_boot[r] = omega_teorico_boot[r] / frq[inv_idx]
        
        sum_w_r = np.sum(w_jueces_boot[r])
        if sum_w_r > 0:
            w_jueces_boot[r] /= sum_w_r
        else:
            w_jueces_boot[r] = 1.0 / m

    # 4. Cálculo final de NO para las 1000 iteraciones en un solo disparo de Numpy
    sims = _no_desde_pesos(f_boot, w_cols=w_jueces_boot)

    return float(no_muestra), float(no_poblacion), float(np.percentile(sims, 2.5)), float(np.percentile(sims, 97.5))