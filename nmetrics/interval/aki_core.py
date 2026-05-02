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
        m_valid = 0 # Contador dinámico de jueces reales en la fila
        
        for val in fila:
            if np.isnan(val): 
                continue # Ignoramos los valores faltantes
            m_valid += 1
            
            if val <= k_min: c[k_min] += 1.0
            elif val >= k_max: c[k_max] += 1.0
            else:
                L, U = math.floor(val), math.ceil(val)
                if L == U: c[L] += 1.0
                else:
                    f = val - L
                    c[U] += f
                    c[L] += (1.0 - f)
        
        df_counts = [c[k] for k in range(k_min, k_max + 1)]
        
        # Una fila necesita al menos 2 jueces para aportar multiplicidad termodinámica
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
# 2. Motor Vectorizado de AKI Clásico (Plano / Sin Pesos / Soporta NaNs)
# ==============================================================================
def _compute_aki_vectorized_bc(X_3d):
    """Calcula AKI asumiendo equiprobabilidad estricta y tolerando valores nulos."""
    S, n, m = X_3d.shape
    m_valid = np.sum(~np.isnan(X_3d), axis=2) # Jueces válidos por sujeto (S x n)
    N_total = np.sum(m_valid, axis=1)         # Total de votos válidos por matriz (S)
    
    # Media global de cada matriz
    row_sums = np.nansum(X_3d, axis=2)
    mean_global = np.nansum(row_sums, axis=1) / N_total 
    
    # Desacuerdo Esperado (De)
    sq_diffs_global = (X_3d - mean_global[:, None, None])**2
    De_num = np.nansum(sq_diffs_global, axis=(1, 2))
    De = De_num / (N_total - 1)
    
    # Desacuerdo Observado (Do)
    with np.errstate(divide='ignore', invalid='ignore'):
        row_means = np.nanmean(X_3d, axis=2, keepdims=True)
        sq_diffs_row = (X_3d - row_means)**2
        Do_num = np.nansum(sq_diffs_row, axis=(1, 2))
        
        # El denominador excluye los -1 solo de las filas que tienen al menos 1 voto
        Do_denom = np.sum(np.maximum(0, m_valid - 1), axis=1)
        Do = Do_num / Do_denom
        
        aki_vals = np.where(De == 0, np.nan, 1 - (Do / De))
        
    return aki_vals

# ==============================================================================
# 3. Cálculo Teórico Poblacional (Matriz Asintótica Masiva)
# ==============================================================================
def calcular_aki_poblacion_asintotica(matriz_entrada, k_min=1, k_max=5, multiplicador=1000):
    """
    Construye el universo termodinámico físico clonando filas y lo evalúa
    usando la fórmula de AKI original pura.
    """
    df_pob, cols_x = calcular_parametros_poblacion_fraccionaria(matriz_entrada, k_min, k_max)
    pesos = df_pob['Num_Tuplas'].values.astype(float)
    
    sum_pesos = np.sum(pesos)
    if sum_pesos == 0: 
        return np.nan # Seguridad por si toda la matriz está vacía o sin consenso posible
        
    prob = pesos / sum_pesos
    
    n, k = matriz_entrada.shape
    N_pop = n * multiplicador
    
    counts = np.round(prob * N_pop).astype(int)
    
    diff = N_pop - np.sum(counts)
    if diff > 0: counts[np.argmax(prob)] += diff
    elif diff < 0: counts[np.argmax(counts)] += diff
        
    X_massive = np.repeat(matriz_entrada, counts, axis=0)
    
    # Evaluar el universo entero con la función pura de AKI
    X_massive_3d = X_massive[None, :, :]
    aki_pob_real = _compute_aki_vectorized_bc(X_massive_3d)[0]
    
    return float(aki_pob_real)

# ==============================================================================
# 4. Cálculo del Coeficiente Muestral y su IC
# ==============================================================================
def calcular_estadisticas_aki(matriz_entrada, S_replicas):
    matriz_empirica = np.array(matriz_entrada, dtype=float)
    n_sujetos, m_evaluadores = matriz_empirica.shape
    
    # 1. Cálculo puntual de AKI para la matriz de muestra
    X_muestra_3d = matriz_empirica[None, :, :]
    aki_muestra = _compute_aki_vectorized_bc(X_muestra_3d)[0]
    
    # 2. Motor de Caos: Bootstrap Clásico (Probabilidad 1/n)
    indices = np.empty((S_replicas, n_sujetos), dtype=int)
    for s in range(S_replicas):
        indices[s] = np.random.choice(n_sujetos, size=n_sujetos, replace=True)
        
    X_3d_bc = matriz_empirica[indices]
    
    # 3. Evaluación vectorizada de réplicas
    aki_replicas = _compute_aki_vectorized_bc(X_3d_bc)
    
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