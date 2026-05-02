import numpy as np
import scipy.stats as stats
import pandas as pd
import math

# ==============================================================================
# 1. Preparar población teórica (Fraccionaria / Difusa)
# ==============================================================================
def calcular_parametros_poblacion_fraccionaria(matriz_entrada, k_min=1, k_max=5):
    n, m = matriz_entrada.shape
    filas_salida = []
    
    for fila in matriz_entrada:
        c = {k: 0.0 for k in range(k_min, k_max + 1)}
        m_valid = 0 # Contador dinámico de jueces reales
        
        for val in fila:
            if np.isnan(val):
                continue # Ignorar valores faltantes
                
            m_valid += 1
            if val <= k_min:
                c[k_min] += 1.0
            elif val >= k_max:
                c[k_max] += 1.0
            else:
                L = math.floor(val)
                U = math.ceil(val)
                if L == U:
                    c[L] += 1.0
                else:
                    f = val - L
                    c[U] += f
                    c[L] += (1.0 - f)
        
        df_counts = [c[k] for k in range(k_min, k_max + 1)]
        
        # Conservación termodinámica (solo filas con >1 juez generan entropía)
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
# 2. Cálculo Teórico Poblacional (Matriz Asintótica)
# ==============================================================================
def calcular_icc_poblacion_asintotica(matriz_entrada, k_min=1, k_max=5, multiplicador=1000):
    df_pob, cols_x = calcular_parametros_poblacion_fraccionaria(matriz_entrada, k_min, k_max)
    pesos = df_pob['Num_Tuplas'].values.astype(float)
    
    sum_pesos = np.sum(pesos)
    if sum_pesos == 0: 
        return np.nan 
        
    prob = pesos / sum_pesos
    n, k = matriz_entrada.shape
    N_pop = n * multiplicador
    
    counts = np.round(prob * N_pop).astype(int)
    
    diff = N_pop - np.sum(counts)
    if diff > 0: counts[np.argmax(prob)] += diff
    elif diff < 0: counts[np.argmax(counts)] += diff
        
    X_massive = np.repeat(matriz_entrada, counts, axis=0)
    stats_pop = calcular_estadisticas_icc21(X_massive)
    return stats_pop['ICC Muestra']

# ==============================================================================
# 3. MOTOR MATEMÁTICO: ICC(2,1) ANOVA ESTÁNDAR (Listwise Deletion)
# ==============================================================================
def calcular_estadisticas_icc21(matriz_entrada, S_replicas=None, alpha=0.05):
    """
    Calcula el estimador puntual del ICC(2,1) replicando el estándar 
    de SPSS/PQStat: Eliminación de filas con NaNs (Listwise Deletion) 
    para mantener el ANOVA ortogonal y balanceado.
    """
    X_raw = np.array(matriz_entrada, dtype=float)
    
    # --- LISTWISE DELETION ---
    # Eliminamos cualquier sujeto que tenga al menos 1 valor NaN
    valid_rows = ~np.isnan(X_raw).any(axis=1)
    X = X_raw[valid_rows]
    
    n, k = X.shape
    
    # Si tras borrar filas no quedan suficientes datos, abortar
    if n < 2 or k < 2:
        return {'ICC Muestra': np.nan, 'IC Inf': np.nan, 'IC Sup': np.nan}

    # Medias Clásicas Balanceadas
    mean_per_subject = X.mean(axis=1)
    mean_per_rater = X.mean(axis=0)
    grand_mean = X.mean()
    
    # Sumas de Cuadrados Clásicas (SST = SSR + SSC + SSE)
    SST = np.sum((X - grand_mean)**2)
    SSR = k * np.sum((mean_per_subject - grand_mean)**2)
    SSC = n * np.sum((mean_per_rater - grand_mean)**2)
    SSE = max(0.0, SST - SSR - SSC) # max() protege contra ruido de punto flotante
    
    # Grados de libertad
    dfR = n - 1
    dfC = k - 1
    dfE = (n - 1) * (k - 1)
    
    # Cuadrados Medios
    MSR = SSR / dfR
    MSC = SSC / dfC
    MSE = SSE / dfE if dfE > 0 else 0
    
    # Fórmula ICC(2,1) McGraw & Wong (1996)
    denom = MSR + (k - 1) * MSE + (k / n) * (MSC - MSE)
    
    if denom == 0 or np.isnan(denom):
        icc_muestra, L, U = np.nan, np.nan, np.nan
    else:
        icc_muestra = (MSR - MSE) / denom
        
        # Constantes de aproximación para el Intervalo de Confianza
        a, b = k / n, (k * n - k - n) / n
        
        v_num = (a * MSC + b * MSE)**2
        v_den = ((a * MSC)**2) / dfC + ((b * MSE)**2) / dfE
        
        if v_den < 1e-15 or np.isnan(v_den):
            L, U = np.nan, np.nan
        else:
            v = v_num / v_den
            F_U = stats.f.ppf(1 - alpha/2, n - 1, v)
            F_L = stats.f.ppf(alpha/2, n - 1, v)
            
            denom_U = F_U * (k * MSC + (k * n - k - n) * MSE) + n * MSR
            denom_L = F_L * (k * MSC + (k * n - k - n) * MSE) + n * MSR
            
            L = n * (MSR - F_U * MSE) / denom_U if denom_U != 0 else np.nan
            U = n * (MSR - F_L * MSE) / denom_L if denom_L != 0 else np.nan
            
            # Ordenar límites si se invierten por denominadores negativos
            if not np.isnan(L) and not np.isnan(U) and L > U:
                L, U = U, L
                
            # Truncar límites al dominio [-1, 1]
            if not np.isnan(L): L = max(-1.0, min(1.0, L))
            if not np.isnan(U): U = max(-1.0, min(1.0, U))

    return {
        'ICC Muestra': icc_muestra,
        'IC Inf': L,
        'IC Sup': U
    }