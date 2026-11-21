import pandas as pd
import numpy as np
import math


# ==============================================================================
# 1. Preparar población teórica (Fraccionaria / Difusa)
# ==============================================================================
def calcular_parametros_poblacion_fraccionaria(matriz_entrada, k_min=1, k_max=5):
    n, m = matriz_entrada.shape
    filas_salida = []

    for fila in matriz_entrada:
        c = {k: 0.0 for k in range(k_min, k_max + 1)}
        m_valid = 0

        for val in fila:
            if np.isnan(val):
                continue
            m_valid += 1

            if val <= k_min:
                c[k_min] += 1.0
            elif val >= k_max:
                c[k_max] += 1.0
            else:
                L, U = math.floor(val), math.ceil(val)
                if L == U:
                    c[L] += 1.0
                else:
                    f = val - L
                    c[U] += f
                    c[L] += (1.0 - f)

        df_counts = [c[k] for k in range(k_min, k_max + 1)]

        if m_valid > 1:
            denominador = 1.0
            for count in df_counts:
                denominador *= math.gamma(count + 1)
            num_tuplas = math.gamma(m_valid + 1) / denominador
        else:
            num_tuplas = 0.0

        filas_salida.append(list(fila) + df_counts + [num_tuplas])

    cols_x = [f"X{i+1}" for i in range(m)]
    cols_df = [f"DF_{k}" for k in range(k_min, k_max + 1)]
    columnas = cols_x + cols_df + ["Num_Tuplas"]
    return pd.DataFrame(filas_salida, columns=columnas), cols_x


# ==============================================================================
# 2. Motor Vectorizado de AKI (Plano / Sin Pesos / Soporta NaNs)
# ==============================================================================
def _compute_aki_vectorized_bc(X_3d, exclude_singletons=True):
    """
    Calcula AKI (alfa de Krippendorff) para escala de intervalo, forma de
    pares/coincidencias, tolerando valores nulos y diseños desbalanceados.

    Parámetros
    ----------
    X_3d : ndarray (S, n, m)
        S matrices de n sujetos x m evaluadores. Puede contener NaN.
    exclude_singletons : bool, default True
        Si True (convención estricta de Krippendorff 2011), las unidades con
        menos de 2 jueces válidos se excluyen por completo del cálculo (no
        entran ni en N, ni en De, ni en Do). Si False, se incluyen en N y De
        pero no en Do (comportamiento del código original).
    """
    S, n, m = X_3d.shape
    m_valid = np.sum(~np.isnan(X_3d), axis=2)  # (S, n)

    # FIX: máscara de unidades que contribuyen al cálculo
    if exclude_singletons:
        valid_units = m_valid >= 2
    else:
        valid_units = m_valid >= 1

    # N_total: total de valores válidos en unidades válidas
    N_total = np.sum(np.where(valid_units, m_valid, 0), axis=1)  # (S,)

    # Enmascarar valores de unidades no válidas
    X_eff = np.where(valid_units[:, :, None], X_3d, np.nan)

    # Media global por matriz (solo valores de unidades válidas)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_global = np.nansum(X_eff, axis=(1, 2)) / N_total  # (S,)

    # Desacuerdo Esperado (De) -- sin cambios respecto al original
    sq_diffs_global = (X_eff - mean_global[:, None, None]) ** 2
    De_num = np.nansum(sq_diffs_global, axis=(1, 2))
    De = np.where(N_total > 1, De_num / (N_total - 1), np.nan)

    # FIX: Desacuerdo Observado (Do) -- forma de pares (Krippendorff)
    # S_u = suma de (x - mean_u)^2 dentro de cada unidad
    row_sums = np.nansum(X_3d, axis=2, keepdims=True)          # (S, n, 1)
    safe_m = np.maximum(m_valid[:, :, None], 1)                # evita div/0
    row_means = row_sums / safe_m                               # (S, n, 1)
    row_means = np.where(m_valid[:, :, None] > 0, row_means, 0.0)

    sq_diffs_row = (X_3d - row_means) ** 2                      # (S, n, m)
    S_u = np.nansum(sq_diffs_row, axis=2)                       # (S, n)

    # Solo unidades con m_u >= 2 aportan a Do (singletons no forman pares)
    contrib_to_Do = valid_units & (m_valid >= 2)
    safe_m_minus_1 = np.maximum(m_valid - 1, 1)
    Do_per_unit = np.where(contrib_to_Do,
                           m_valid * S_u / safe_m_minus_1,
                           0.0)
    Do_num = np.sum(Do_per_unit, axis=1)                        # (S,)
    Do = np.where(N_total > 0, Do_num / N_total, np.nan)

    with np.errstate(invalid='ignore', divide='ignore'):
        aki_vals = np.where(De == 0, np.nan, 1 - Do / De)

    return aki_vals


# ==============================================================================
# 3. Cálculo Teórico Poblacional (Matriz Asintótica Masiva con N_pop Adaptativo)
# ==============================================================================
def calcular_aki_poblacion_asintotica(matriz_entrada, k_min=1, k_max=5,
                                      multiplicador=None,
                                      exclude_singletons=True):
    """
    Construye el universo termodinámico físico clonando filas y lo evalúa
    usando la fórmula de AKI original pura.
    """
    df_pob, cols_x = calcular_parametros_poblacion_fraccionaria(
        matriz_entrada, k_min, k_max)
    pesos = df_pob['Num_Tuplas'].values.astype(float)

    sum_pesos = np.sum(pesos)
    if sum_pesos == 0:
        return np.nan

    prob = pesos / sum_pesos
    n, k = matriz_entrada.shape

    # --- ESCALADO ADAPTATIVO DE POBLACIÓN OBJETIVO ---
    if multiplicador is None:
        N_pop = int(np.clip(n * 1000, 100_000, 500_000))
    else:
        N_pop = int(n * multiplicador)

    counts = np.round(prob * N_pop).astype(int)

    # Ajuste de diferencias por redondeo
    diff = N_pop - np.sum(counts)
    if diff > 0:
        counts[np.argmax(prob)] += diff
    elif diff < 0:
        counts[np.argmax(counts)] += diff

    counts = np.maximum(0, counts).astype(int)

    X_massive = np.repeat(matriz_entrada, counts, axis=0)

    # Evaluar el universo entero con la función pura de AKI
    X_massive_3d = X_massive[None, :, :]
    aki_pob_real = _compute_aki_vectorized_bc(
        X_massive_3d, exclude_singletons=exclude_singletons)[0]

    return float(aki_pob_real)


# ==============================================================================
# 4. Cálculo del Coeficiente Muestral y su IC
# ==============================================================================
def calcular_estadisticas_aki(matriz_entrada, S_replicas, exclude_singletons=True):
    matriz_empirica = np.array(matriz_entrada, dtype=float)
    n_sujetos, m_evaluadores = matriz_empirica.shape

    # 1. Cálculo puntual de AKI para la matriz de muestra
    X_muestra_3d = matriz_empirica[None, :, :]
    aki_muestra = _compute_aki_vectorized_bc(
        X_muestra_3d, exclude_singletons=exclude_singletons)[0]

    # 2. Motor de Caos: Bootstrap Clásico (Probabilidad 1/n)
    indices = np.empty((S_replicas, n_sujetos), dtype=int)
    for s in range(S_replicas):
        indices[s] = np.random.choice(n_sujetos, size=n_sujetos, replace=True)

    X_3d_bc = matriz_empirica[indices]

    # 3. Evaluación vectorizada de réplicas
    aki_replicas = _compute_aki_vectorized_bc(
        X_3d_bc, exclude_singletons=exclude_singletons)

    # 4. Cálculo de los percentiles (IC)
    clean_aki = aki_replicas[~np.isnan(aki_replicas)]
    if len(clean_aki) > 0:
        ic_inf, ic_sup = np.percentile(clean_aki, [2.5, 97.5])
    else:
        ic_inf, ic_sup = np.nan, np.nan

    return {
        'AKI Muestra': aki_muestra,
        'IC Inf': ic_inf,
        'IC Sup': ic_sup
    }