import math
from collections import Counter, defaultdict
from functools import lru_cache
import numpy as np
import pandas as pd


# ==============================================================================
# 1. ESCÁNER DE ANOMALÍAS ORDINALES (Entrelazamiento Vertical)
# ==============================================================================
def detectar_anomalias_no(matriz_entrada, k_escala=5, umbral_sigma=1.0):
    """Detecta jueces (columnas) que introducen entropía vertical anómala

    midiendo la desviación de sus acuerdos cruzados mediante Ranking Denso.
    """
    X = np.array(matriz_entrada, dtype=float)
    X_cat = np.floor(X + 0.5)
    X_cat = np.clip(X_cat, 1, k_escala)
    n, m = X.shape

    # Conversión topológica a rangos densos por columna
    X_sm = X_cat.T
    sort_idx = np.argsort(X_sm, axis=1)
    row_idx = np.arange(m)[:, None]
    sorted_X = X_sm[row_idx, sort_idx]

    diffs = np.diff(sorted_X, axis=1)
    diffs[np.isnan(diffs)] = 0.0
    step = np.concatenate([np.zeros((m, 1)), (diffs != 0).astype(int)], axis=1)
    ranks = np.cumsum(step, axis=1) + 1

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

    acuerdo_local = np.divide(
        coincidencias_juez,
        emparejamientos_juez,
        out=np.full_like(coincidencias_juez, np.nan, dtype=float),
        where=emparejamientos_juez > 0,
    )

    valid = ~np.isnan(acuerdo_local)
    if not np.any(valid):
        return pd.DataFrame(), np.nan, np.nan, np.nan

    mu_global = np.mean(acuerdo_local[valid])
    sigma_global = np.std(acuerdo_local[valid])
    limite = mu_global - (umbral_sigma * sigma_global)
    anomalo = (acuerdo_local < limite) & valid

    df_anomalias = pd.DataFrame(
        {
            "Juez_ID": np.arange(1, m + 1),
            "Acuerdo_Local": acuerdo_local,
            "Es_Anomalo": anomalo,
        }
    )
    return df_anomalias, mu_global, sigma_global, limite


# ==============================================================================
# 2. MOTOR DE CONSTRUCCIÓN DE CE (Versión Sincronizada con la Hoja)
# ==============================================================================
@lru_cache(maxsize=128)
def _build_macrostate_dictionary_no(m_jueces, k_escala):
    """Genera el diccionario de multiplicidades (Omega) adaptado a la fórmula de

    la hoja. Garantiza consistencia absoluta entre la simulación teórica y la
    empírica.
    """
    macro_dict = {}

    def _get_bounded_partitions(n, max_len, max_val=None):
        if max_val is None:
            max_val = n
        if n == 0:
            yield ()
        if max_len == 0:
            return
        for i in range(min(n, max_val), 0, -1):
            for p in _get_bounded_partitions(n - i, max_len - 1, i):
                yield (i,) + p

    for p in _get_bounded_partitions(m_jueces, k_escala):
        b = len(p)
        factor_cat = float(math.perm(k_escala, b))

        factor_ens = 1.0
        for s, c in Counter(p).items():
            factor_ens *= math.factorial(s * c) / (
                (math.factorial(s) ** c) * math.factorial(c)
            )

        omega_ce = factor_cat * factor_ens
        macro_dict[tuple(sorted(p, reverse=True))] = omega_ce

    return macro_dict


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


def analizar_termodinamica_no(
    n_sujetos, m_jueces, k_escala, valor_observado=None
):
    """Distribución Acumulada Discreta basada en el colapso exacto del

    hiperespacio.
    """
    azar_esperado = float(np.sqrt(1.0 / k_escala))
    min_N = calcular_suelo_cristal_no(m_jueces, k_escala)

    prob_por_no = defaultdict(float)
    p_base = (1.0 / k_escala) ** m_jueces
    m_pairs = m_jueces * (m_jueces - 1) / 2.0

    macro_dict = _build_macrostate_dictionary_no(m_jueces, k_escala)

    for p, omega_ce in macro_dict.items():
        c_p = sum((c * (c - 1)) / 2.0 for c in p)
        mu_macro = c_p / m_pairs if m_pairs > 0 else 0.0
        no_macro = np.sqrt(mu_macro)

        p_total = p_base * omega_ce
        prob_por_no[no_macro] += p_total

    no_ordenados = np.array(sorted(prob_por_no.keys()))
    probs = np.array([prob_por_no[v] for v in no_ordenados])
    probs = probs / np.sum(probs)
    cdf = np.cumsum(probs)

    def _interpolar_masa(valor):
        if valor <= no_ordenados[0]:
            return cdf[0] * 100.0 if valor == no_ordenados[0] else 0.0
        if valor >= no_ordenados[-1]:
            return 100.0
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
    percentil_obs = (
        _interpolar_masa(valor_observado) if valor_observado is not None else 0.0
    )

    info_estructura = {
        "Percentil_Azar_Real": percentil_azar_exacto,
        "Suelo": min_N,
    }
    return (
        float(azar_esperado),
        info_estructura,
        float(percentil_obs),
        float(min_N),
    )


# ==============================================================================
# 3. MOTOR DE INFERENCIA UNIFICADA (Dinámico y Optimizado)
# ==============================================================================
def calcular_estadisticas_no_unificada(dict_estados, k_escala, replicas=1000):
    estados_lista = list(dict_estados.keys())
    f_t = np.array(list(dict_estados.values()), dtype=float)
    n_total = int(np.sum(f_t))

    X_estados = np.array([list(e) for e in estados_lista], dtype=float)
    U, m = X_estados.shape
    m_valid = np.sum(~np.isnan(X_estados), axis=1)

    # 1. Ranking Denso por COLUMNA (Vertical, por Juez)
    R_estados = np.zeros_like(X_estados, dtype=float)
    for j in range(m):
        col = X_estados[:, j]
        mask = ~np.isnan(col)
        valid = col[mask]
        if len(valid) > 0:
            uniq = np.sort(np.unique(valid))
            mapping = {val: idx + 1 for idx, val in enumerate(uniq)}
            R_estados[mask, j] = [mapping[v] for v in valid]
        R_estados[~mask, j] = np.nan

    # 2. Coincidencias Observadas sobre Rangos Densos
    match_u_j = np.zeros((U, m), dtype=float)
    for j in range(m):
        val_j = R_estados[:, j]
        valid_j = ~np.isnan(val_j)
        for k in range(m):
            if j == k:
                continue
            val_k = R_estados[:, k]
            valid_k = ~np.isnan(val_k)
            both_valid = valid_j & valid_k
            match_u_j[:, j] += (both_valid & (val_j == val_k)).astype(float)

    # 3. Optimización Vectorial: Coincidencias Máximas Dinámicas por fila
    valid_mask = ~np.isnan(X_estados)
    pos_u_j = np.where(valid_mask, (m_valid[:, None] - 1), 0.0)

    def _no_desde_pesos(w_rows):
        if w_rows.ndim == 1:
            w_rows = w_rows[None, :]
        C_j = np.dot(w_rows, match_u_j)
        P_j = np.dot(w_rows, pos_u_j)
        with np.errstate(divide="ignore", invalid="ignore"):
            A_j = np.where(P_j > 0, C_j / P_j, np.nan)
        mu = np.nanmean(A_j, axis=1)
        sigma = np.nanstd(A_j, axis=1)
        return np.sqrt(np.maximum(mu * (1.0 - sigma), 0.0))

    # ---------------------------------------------------------
    # NO MUESTRA
    # ---------------------------------------------------------
    w_muestra = f_t / n_total
    w_muestra = w_muestra / np.sum(w_muestra)
    w_muestra[-1] = 1.0 - np.sum(w_muestra[:-1])
    w_muestra = np.clip(w_muestra, 0.0, 1.0)

    no_muestra = _no_desde_pesos(w_muestra)[0]

    # ---------------------------------------------------------
    # NO POBLACIÓN (Extracción de firmas e hiperespacio)
    # ---------------------------------------------------------
    firmas = []
    for i in range(U):
        row = R_estados[i]
        vals = row[~np.isnan(row)]
        _, counts = np.unique(vals, return_counts=True)
        firmas.append(tuple(sorted(counts, reverse=True)))

    omega_hoja = np.zeros(U, dtype=float)
    for i, f in enumerate(firmas):
        b = len(f)
        factor_cat = float(math.perm(k_escala, b))
        factor_ens = 1.0
        for s, c in Counter(f).items():
            factor_ens *= math.factorial(s * c) / (
                (math.factorial(s) ** c) * math.factorial(c)
            )
        omega_hoja[i] = factor_cat * factor_ens

    firmas_str = [str(f) for f in firmas]
    unique_firmas, inverse_idx = np.unique(firmas_str, return_inverse=True)

    # CRÍTICO: Restitución de la Clausura del Espacio Completo Uniforme.
    # Si U es exactamente k^m, la muestra ya contiene el universo expandido físicamente;
    # no requiere proyección expansiva (evita elevar Omega al cuadrado).
    if U == int(k_escala**m):
        w_pob = f_t / np.sum(f_t)
    else:
        w_pob = (f_t * omega_hoja) / np.sum(f_t * omega_hoja)

    no_poblacion = _no_desde_pesos(w_pob)[0]

    # ---------------------------------------------------------
    # BOOTSTRAP COMBINATORIO
    # ---------------------------------------------------------
    safe_n = max(n_total, 2)
    p_boot = f_t / n_total
    p_boot[-1] = 1.0 - np.sum(p_boot[:-1])
    p_boot = np.clip(p_boot, 0.0, 1.0)

    f_boot = np.random.multinomial(safe_n, p_boot, size=replicas)
    w_boot = np.zeros((replicas, U))

    for r in range(replicas):
        f_M_r = np.zeros(len(unique_firmas))
        np.add.at(f_M_r, inverse_idx, f_boot[r])
        f_M_mapped_r = f_M_r[inverse_idx]

        validez_mask = (m_valid > 1) & (f_M_mapped_r > 0)
        w_r = np.zeros_like(f_boot[r], dtype=float)

        w_r[validez_mask] = (
            f_boot[r, validez_mask]
            * np.divide(
                omega_hoja,
                f_M_mapped_r,
                out=np.zeros_like(f_M_mapped_r),
                where=f_M_mapped_r != 0,
            )[validez_mask]
        )

        sum_w_r = np.sum(w_r)
        if sum_w_r > 0:
            w_r = w_r / sum_w_r
        else:
            w_r = np.ones_like(w_r) / len(w_r)

        w_r[-1] = 1.0 - np.sum(w_r[:-1])
        w_boot[r] = np.clip(w_r, 0.0, 1.0)

    sims = _no_desde_pesos(w_boot)

    return (
        float(no_muestra),
        float(no_poblacion),
        float(np.percentile(sims, 2.5)),
        float(np.percentile(sims, 97.5)),
    )