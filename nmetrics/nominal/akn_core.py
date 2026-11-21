import pandas as pd
import numpy as np
from math import factorial
from collections import Counter


# ==============================================================================
# 1. Preparar población teórica (Espacio de Configuración Nominal)
# ==============================================================================
def calcular_parametros_poblacion_nominal(matriz_entrada, k_escala):
    n, m_total = matriz_entrada.shape
    filas_salida = []
    fact_k = factorial(k_escala)

    for fila in matriz_entrada:
        m_valid = int(np.sum(~np.isnan(fila)))
        df_counts = [int(np.sum(fila == v)) for v in range(1, k_escala + 1)]

        # 1. Permutaciones de jueces válidos (ignorando NaN)
        if m_valid > 0:
            fact_m = factorial(m_valid)
            denominador_m = 1
            for c in df_counts:
                denominador_m *= factorial(c)
            tuplas_evals = fact_m // denominador_m
        else:
            tuplas_evals = 0

        # 2. Permutaciones de la escala
        counts_of_counts = Counter(df_counts)
        denominador_k = 1
        for freq in counts_of_counts.values():
            denominador_k *= factorial(freq)
        df_val = fact_k // denominador_k

        num_tuplas = df_val * tuplas_evals
        filas_salida.append(list(fila) + [num_tuplas])

    cols_x = [f"X{i+1}" for i in range(m_total)]
    columnas = cols_x + ["Num_Tuplas"]
    return pd.DataFrame(filas_salida, columns=columnas), cols_x


# ==============================================================================
# 2. Motor Vectorizado de AKN (Nominal) — Do/De correctos, soporta NaN
# ==============================================================================
def _compute_akn_vectorized_bc(X_3d, k_escala=5, exclude_singletons=True):
    """
    Alpha de Krippendorff nominal con datos faltantes y diseño desbalanceado.

    Do = (1/N) * Σ_u [ (m_u(m_u-1) - Σ_c n_cu(n_cu-1)) / (m_u-1) ]
    De = 1 - Pe
    Pe = [ Σ_c n_c(n_c-1) ] / [ N(N-1) ]
    N  = Σ_u m_u
    """
    S, n, m = X_3d.shape
    m_valid = np.sum(~np.isnan(X_3d), axis=2)  # (S, n)

    if exclude_singletons:
        valid_units = m_valid >= 2
    else:
        valid_units = m_valid >= 1

    X_eff = np.where(valid_units[:, :, None], X_3d, np.nan)

    all_valid = X_eff[~np.isnan(X_eff)]
    if all_valid.size == 0:
        return np.full(S, np.nan)

    unique_vals = np.unique(all_valid)
    counts = np.zeros((S, n, len(unique_vals)))

    for idx, val in enumerate(unique_vals):
        counts[:, :, idx] = np.sum(X_eff == val, axis=2)

    # N_total = Σ m_u sobre unidades válidas
    N_total = np.sum(np.where(valid_units, m_valid, 0), axis=1)

    # Pares concordantes dentro de cada unidad
    A_u = np.sum(counts * (counts - 1), axis=2)  # (S, n)

    # Pares discordantes dentro de cada unidad
    disagree_pairs = m_valid * (m_valid - 1) - A_u

    contrib_to_Do = valid_units & (m_valid >= 2)
    safe_m_minus_1 = np.maximum(m_valid - 1, 1)

    Do_per_unit = np.where(
        contrib_to_Do,
        disagree_pairs / safe_m_minus_1,
        0.0
    )
    Do_num = np.sum(Do_per_unit, axis=1)

    with np.errstate(invalid='ignore', divide='ignore'):
        Do = np.where(N_total > 0, Do_num / N_total, np.nan)

    # Acuerdo esperado Pe
    total_counts = np.sum(counts, axis=1)  # (S, k)
    agree_pairs_total = np.sum(total_counts * (total_counts - 1), axis=1)

    with np.errstate(invalid='ignore', divide='ignore'):
        Pe = np.where(
            N_total > 1,
            agree_pairs_total / (N_total * (N_total - 1)),
            np.nan
        )
        De = 1.0 - Pe
        akn_vals = np.where(De == 0, np.nan, 1.0 - Do / De)

    return akn_vals


# ==============================================================================
# 3. Cálculo Teórico Poblacional
# ==============================================================================
def calcular_akn_poblacion_asintotica(matriz_entrada, k_escala=5,
                                      multiplicador=None,
                                      exclude_singletons=True):
    df_pob, cols_x = calcular_parametros_poblacion_nominal(
        matriz_entrada, k_escala
    )
    pesos = df_pob['Num_Tuplas'].values.astype(float)

    sum_pesos = np.sum(pesos)
    if sum_pesos == 0:
        return np.nan

    prob = pesos / sum_pesos
    n, k = matriz_entrada.shape

    if multiplicador is None:
        N_pop = int(np.clip(n * 1000, 100_000, 500_000))
    else:
        N_pop = int(n * multiplicador)

    counts = np.round(prob * N_pop).astype(int)

    diff = N_pop - np.sum(counts)
    if diff > 0:
        counts[np.argmax(prob)] += diff
    elif diff < 0:
        counts[np.argmax(counts)] += diff

    counts = np.maximum(0, counts).astype(int)

    X_massive = np.repeat(matriz_entrada, counts, axis=0)
    X_massive_3d = X_massive[None, :, :]

    akn_pob_real = _compute_akn_vectorized_bc(
        X_massive_3d, k_escala,
        exclude_singletons=exclude_singletons
    )[0]

    return float(akn_pob_real)


# ==============================================================================
# 4. Cálculo del Coeficiente Muestral y su IC
# ==============================================================================
# def calcular_estadisticas_akn(matriz_entrada, S_replicas, k_escala=5,
                              # exclude_singletons=True):
    # matriz_empirica = np.array(matriz_entrada, dtype=float)
    # n_sujetos, m_evaluadores = matriz_empirica.shape

    # X_muestra_3d = matriz_empirica[None, :, :]
    # akn_muestra = _compute_akn_vectorized_bc(
        # X_muestra_3d, k_escala,
        # exclude_singletons=exclude_singletons
    # )[0]

    # indices = np.empty((S_replicas, n_sujetos), dtype=int)
    # for s in range(S_replicas):
        # indices[s] = np.random.choice(n_sujetos, size=n_sujetos, replace=True)

    # X_3d_bc = matriz_empirica[indices]

    # akn_replicas = _compute_akn_vectorized_bc(
        # X_3d_bc, k_escala,
        # exclude_singletons=exclude_singletons
    # )

    # clean_akn = akn_replicas[~np.isnan(akn_replicas)]
    # if len(clean_akn) > 0:
        # ic_inf, ic_sup = np.percentile(clean_akn, [2.5, 97.5])
    # else:
        # ic_inf, ic_sup = np.nan, np.nan

    # return {
        # 'AKN Muestra': akn_muestra,
        # 'IC Inf': ic_inf,
        # 'IC Sup': ic_sup
    # }
def calcular_estadisticas_akn(matriz_entrada, S_replicas, k_escala=28,
                              exclude_singletons=True):
    import gc
    matriz_empirica = np.array(matriz_entrada, dtype=float)
    n_sujetos, m_evaluadores = matriz_empirica.shape

    # 1. Valor puntual (usa toda la matriz, es rápido)
    X_muestra_3d = matriz_empirica[None, :, :]
    akn_muestra = _compute_akn_vectorized_bc(
        X_muestra_3d, k_escala,
        exclude_singletons=exclude_singletons
    )[0]

    # 2. Límite de memoria muy conservador (5 millones de elementos ~ 40 MB)
    # Esto obliga a procesar de 1 a 5 réplicas a la vez.
    MAX_ELEMENTOS = 5_000_000
    
    # Calcular el tamaño del lote
    elementos_por_replica = n_sujetos * m_evaluadores
    chunk_size = max(1, int(MAX_ELEMENTOS / elementos_por_replica))

    akn_replicas_lista = []

    # 3. Bucle de bootstrap por lotes
    for i in range(0, S_replicas, chunk_size):
        s_chunk = min(chunk_size, S_replicas - i)
        
        # Generar remuestreo
        indices = np.random.choice(n_sujetos, size=(s_chunk, n_sujetos), replace=True)
        X_3d_bc = matriz_empirica[indices]
        
        # Calcular AKN para este lote (memoria controlada)
        res_chunk = _compute_akn_vectorized_bc(
            X_3d_bc, k_escala,
            exclude_singletons=exclude_singletons
        )
        akn_replicas_lista.append(res_chunk)
        
        # 🧹 Forzar recolección de basura
        del X_3d_bc, indices, res_chunk
        gc.collect()

    # 4. Unir resultados y calcular IC
    akn_replicas = np.concatenate(akn_replicas_lista)
    clean_akn = akn_replicas[~np.isnan(akn_replicas)]
    
    if len(clean_akn) > 0:
        ic_inf, ic_sup = np.percentile(clean_akn, [2.5, 97.5])
    else:
        ic_inf, ic_sup = np.nan, np.nan

    return {
        'AKN Muestra': akn_muestra,
        'IC Inf': ic_inf,
        'IC Sup': ic_sup
    }