import numpy as np
import pandas as pd
from math import comb


# ==============================================================================
# 2. Motor Vectorizado de AKO (Ordinal) — forma de pares, soporta NaN
# ==============================================================================
def _compute_ako_vectorized(X_3d, k_escala=None, exclude_singletons=True):
    """
    Calcula el Alpha Ordinal de Krippendorff tolerando NaN y diseño
    desbalanceado.

    Formulación (Krippendorff 2011, escala ordinal):
        δ²(c,k) = [ Σ_{g=c}^{k} n_g - n_c/2 - n_k/2 ]²
        Do = (1/N) * Σ_u [ Σ_{c,k} n_cu n_ku δ²(c,k) / (m_u - 1) ]
        De = [ Σ_{c,k} n_c n_k δ²(c,k) ] / [ N (N - 1) ]

    Parámetros
    ----------
    X_3d : ndarray (S, n, m)
    k_escala : int o None (no usado; se conserva por compatibilidad)
    exclude_singletons : bool, default True
        Si True, unidades con m_u < 2 quedan excluidas de N, De y Do.
        Si False, se incluyen en N y De pero no en Do.
    """
    S, n, m = X_3d.shape
    m_valid = np.sum(~np.isnan(X_3d), axis=2)  # (S, n)

    # FIX: máscara de unidades que contribuyen al cálculo
    if exclude_singletons:
        valid_units = m_valid >= 2
    else:
        valid_units = m_valid >= 1

    X_eff = np.where(valid_units[:, :, None], X_3d, np.nan)

    # Categorías existentes (ordenadas)
    all_valid = X_eff[~np.isnan(X_eff)]
    if all_valid.size == 0:
        return np.full(S, np.nan)
    unique_vals = np.sort(np.unique(all_valid))
    k = len(unique_vals)

    freqs = np.zeros((S, k))            # (S, k)
    row_freqs = np.zeros((S, n, k))     # (S, n, k)

    for v, val in enumerate(unique_vals):
        mask = (X_eff == val)
        freqs[:, v] = np.sum(mask, axis=(1, 2))
        row_freqs[:, :, v] = np.sum(mask, axis=2)

    # Distancia ordinal basada en marginales globales
    D = np.zeros((S, k, k))
    for c in range(k):
        for k_val in range(c + 1, k):
            s_val = np.sum(freqs[:, c:k_val + 1], axis=1)
            d = s_val - freqs[:, c] / 2.0 - freqs[:, k_val] / 2.0
            D[:, c, k_val] = d ** 2
            D[:, k_val, c] = d ** 2

    # N_total por matriz
    N_total = np.sum(np.where(valid_units, m_valid, 0), axis=1)  # (S,)

    # FIX: De normalizado con N(N-1)
    De_num = np.einsum('si,sj,sij->s', freqs, freqs, D)  # (S,)
    with np.errstate(invalid='ignore', divide='ignore'):
        De = np.where(N_total > 1,
                      De_num / (N_total * (N_total - 1)),
                      np.nan)

    # FIX: Do normalizado por (m_u - 1) y por N
    contrib_to_Do = valid_units & (m_valid >= 2)
    safe_m_minus_1 = np.maximum(m_valid - 1, 1)            # (S, n)
    pair_dist_per_unit = np.einsum('snc,snk,sck->sn',
                                   row_freqs, row_freqs, D)  # (S, n)
    Do_per_unit = np.where(contrib_to_Do,
                           pair_dist_per_unit / safe_m_minus_1,
                           0.0)
    Do_num = np.sum(Do_per_unit, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        Do = np.where(N_total > 0, Do_num / N_total, np.nan)

    with np.errstate(invalid='ignore', divide='ignore'):
        ako = np.where(De == 0, np.nan, 1.0 - Do / De)

    return ako


# ==============================================================================
# 3. Cálculo Teórico Poblacional (Matriz Asintótica con M_pop Adaptativo)
# ==============================================================================
def calcular_ako_poblacion_asintotica(matriz_entrada, k_escala=5,
                                      multiplicador=None,
                                      exclude_singletons=True):
    n, m = matriz_entrada.shape
    c_arr = np.zeros(m, dtype=int)
    for val in range(1, k_escala + 1):
        c_arr += (matriz_entrada == val).any(axis=0).astype(int)

    pesos = np.array([comb(k_escala, c) for c in c_arr], dtype=float)
    sum_pesos = np.sum(pesos)
    if sum_pesos == 0:
        return np.nan

    prob = pesos / sum_pesos

    # --- ESCALADO ADAPTATIVO DE EVALUADORES (COLUMNAS) ---
    if multiplicador is None:
        M_pop = int(np.clip(m * 100, 100, 1_000))
    else:
        M_pop = int(m * multiplicador)

    counts = np.round(prob * M_pop).astype(int)

    diff = M_pop - np.sum(counts)
    if diff > 0:
        counts[np.argmax(prob)] += diff
    elif diff < 0:
        counts[np.argmax(counts)] += diff

    counts = np.maximum(0, counts).astype(int)

    X_massive = np.repeat(matriz_entrada, counts, axis=1)
    ako_pob_real = _compute_ako_vectorized(
        X_massive[None, :, :], k_escala,
        exclude_singletons=exclude_singletons)[0]
    return float(ako_pob_real)


# ==============================================================================
# 4. Cálculo del Coeficiente Muestral y su IC
# ==============================================================================
def calcular_estadisticas_ako(matriz_entrada, S_replicas, k_escala=5,
                              exclude_singletons=True):
    matriz_empirica = np.array(matriz_entrada, dtype=float)
    n_sujetos, m_evaluadores = matriz_empirica.shape

    X_muestra_3d = matriz_empirica[None, :, :]
    ako_muestra = _compute_ako_vectorized(
        X_muestra_3d, k_escala,
        exclude_singletons=exclude_singletons)[0]

    # Bootstrap Clásico sobre filas (sujetos)
    indices = np.empty((S_replicas, n_sujetos), dtype=int)
    for s in range(S_replicas):
        indices[s] = np.random.choice(n_sujetos, size=n_sujetos, replace=True)

    X_3d_bc = matriz_empirica[indices]
    ako_replicas = _compute_ako_vectorized(
        X_3d_bc, k_escala,
        exclude_singletons=exclude_singletons)

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