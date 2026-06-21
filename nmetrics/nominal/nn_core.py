import pandas as pd
import numpy as np
import math
import itertools
from collections import defaultdict
from functools import lru_cache
from scipy.special import gammaln

# ==============================================================================
# 1. CONSTRUCCIÓN DEL HIPERESPACIO Y MACROESTADOS (NIVEL III)
# ==============================================================================
# ==============================================================================
# REEMPLAZO EN nn_core.py: LA VÍA DE LAS PARTICIONES ENTERAS (O(P(m)))
# ==============================================================================
@lru_cache(maxsize=128)
def _build_macrostate_dictionary_nn(m_jueces, k_escala):
    fact = [math.factorial(i) for i in range(max(m_jueces, k_escala) + 1)]
    max_coincidencias = (m_jueces * (m_jueces - 1)) / 2
    macro_dict = defaultdict(float)
    
    # 1. Generador ultra-rápido LIMITADO a k_escala (Evita explosión combinatoria)
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
        
        # A) Cálculo de coincidencias (mismo nivel de acuerdo para toda la clase)
        coincidencias = sum((c * (c - 1)) >> 1 for c in p)
        acuerdo = round(coincidencias / max_coincidencias, 8) if max_coincidencias > 0 else 0.0
        
        # B) Multiplicidad base (permutaciones de los jueces dentro de los bloques)
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
            
        # D) Multiplicidad total de la clase de equivalencia
        multiplicidad_total = multiplicidad_base * peso_escala
        macro_dict[acuerdo] += multiplicidad_total
        
    return dict(macro_dict)

# ==============================================================================
# 2. MOTOR TERMODINÁMICO DE SIMULACIÓN PONDERADA
# ==============================================================================

def detectar_anomalias_nn(matriz_entrada, k_escala=5, umbral_sigma=1.0):
    X = np.array(matriz_entrada, dtype=float)
    m_valid = np.sum(~np.isnan(X), axis=1)
    
    counts = np.zeros((len(X), k_escala))
    X_clip = np.floor(X + 0.5); X_clip = np.clip(X_clip, 1, k_escala)
    for k_val in range(1, k_escala + 1):
        counts[:, k_val-1] = np.sum(X_clip == k_val, axis=1)
        
    coincidencias_row = np.sum(counts * (counts - 1) / 2, axis=1)
    max_coincidencias = m_valid * (m_valid - 1) / 2
    
    acuerdo_local = np.divide(coincidencias_row, max_coincidencias,
                              out=np.full_like(coincidencias_row, np.nan, dtype=float),
                              where=max_coincidencias > 0)
    
    valid = ~np.isnan(acuerdo_local)
    if not np.any(valid): return pd.DataFrame(), np.nan, np.nan, np.nan
    
    mu_global = np.mean(acuerdo_local[valid])
    sigma_global = np.std(acuerdo_local[valid])
    limite = mu_global - (umbral_sigma * sigma_global)
    anomalo = (acuerdo_local < limite) & valid
    df_anomalias = pd.DataFrame({'Sujeto_ID': np.arange(1, len(X) + 1), 'Acuerdo_Local': acuerdo_local, 'Es_Anomalo': anomalo})
    return df_anomalias, mu_global, sigma_global, limite

def calcular_azar_termodinamico_nn(m_jueces, k_escala=5):
    macro_dict = _build_macrostate_dictionary_nn(m_jueces, k_escala)
    acuerdos = np.array(list(macro_dict.keys()))
    w_topologico = np.array(list(macro_dict.values()))
    
    sum_w = np.sum(w_topologico)
    mu_global = np.sum(w_topologico * acuerdos) / sum_w
    var_global = np.sum(w_topologico * (acuerdos - mu_global)**2) / sum_w
    sigma_global = np.sqrt(var_global)
    
    return np.sqrt(max(0.0, mu_global * (1.0 - sigma_global)))

def calcular_percentil_universal_nn(nn_empirico, m_jueces, k_escala=5):
    macro_dict = _build_macrostate_dictionary_nn(m_jueces, k_escala)
    agrupados = defaultdict(float)
    total_espacio = sum(macro_dict.values()) 
    
    for acuerdo, multiplicidad_ce in macro_dict.items():
        nn_val = np.sqrt(acuerdo)
        agrupados[round(nn_val, 6)] += multiplicidad_ce
        
    ordenados = sorted(agrupados.items())
    percentiles = []
    acumulado = 0.0
    for nn_val, mult in ordenados:
        acumulado += mult
        percentil_sup = acumulado / total_espacio
        percentiles.append({'nn': nn_val, 'p_sup': percentil_sup})
        
    nn_lower = None; p_lower = None
    nn_upper = None; p_upper = None
    
    for p in percentiles:
        if p['nn'] <= nn_empirico:
            nn_lower = p['nn']; p_lower = p['p_sup']
        else:
            break  # 🚀 VITAL: Romper cuando superamos el límite inferior
            
    for p in reversed(percentiles):
        if p['nn'] >= nn_empirico:
            nn_upper = p['nn']; p_upper = p['p_sup']
        else:
            break  # 🚀 VITAL: Romper cuando bajamos del límite superior
            
    if nn_lower is None: return 0.0
    if nn_upper is None: return 100.0
    if nn_lower == nn_upper: return p_lower * 100.0
    
    p_interp = p_lower + (p_upper - p_lower) * (nn_empirico - nn_lower) / (nn_upper - nn_lower)
    return p_interp * 100.0
    
    
def calcular_poblacion_real_teorica(m_jueces, k_escala=5):
    """
    Calcula el NN de equilibrio termodinámico (Entropía máxima).
    Este valor es una constante del sistema para una configuración (m, k).
    """
    macro_dict = _build_macrostate_dictionary_nn(m_jueces, k_escala)
    acuerdos = np.array(list(macro_dict.keys()))
    multiplicidades = np.array(list(macro_dict.values()))
    
    mu_poblacional = np.sum(acuerdos * multiplicidades) / np.sum(multiplicidades)
    return float(np.sqrt(mu_poblacional))

# ==============================================================================
# MOTOR DE INFERENCIA UNIFICADA (Producto Punto - Big Data Ready)
# ==============================================================================
def calcular_estadisticas_nn_unificada(dict_estados, k_escala, replicas=1000):
    estados_lista = list(dict_estados.keys())
    f_t = np.array(list(dict_estados.values()), dtype=float)
    n_total = int(np.sum(f_t))
    
    try:
        X_estados = np.array([list(e) for e in estados_lista], dtype=float)
    except ValueError:
        raise ValueError("La matriz contiene datos no numéricos.")

    U, m = X_estados.shape

    # ---------------------------------------------------------
    # PRE-CÁLCULO DEL ACUERDO ESTATAL NOMINAL (1 Sola Vez)
    # ---------------------------------------------------------
    m_valid = np.sum(~np.isnan(X_estados), axis=1)
    max_coincidencias = m_valid * (m_valid - 1) / 2

    # ---------------------------------------------------------
    # CONTEO TOPOLÓGICO AGNÓSTICO (Inmune a etiquetas absolutas)
    # ---------------------------------------------------------
    counts = np.zeros((U, k_escala))
    X_clean = np.floor(X_estados + 0.5)
    
    for i in range(U):
        # Ignoramos los NaN de esta fila
        row_valid = X_clean[i][~np.isnan(X_clean[i])]
        if len(row_valid) > 0:
            # np.unique cuenta la frecuencia de cada bloque, sin importar si la etiqueta es 1, 10 o 500
            _, freqs = np.unique(row_valid, return_counts=True)
            # Rellenamos el vector counts por la izquierda (el orden no importa para NN)
            counts[i, :len(freqs)] = freqs

    coincidencias = np.sum(counts * (counts - 1) / 2, axis=1)

    acuerdo = np.divide(coincidencias, max_coincidencias,
                        out=np.zeros_like(coincidencias, dtype=float),
                        where=max_coincidencias > 0)
    acuerdo = np.nan_to_num(acuerdo, nan=0.0)

    # Motor Lambda ultrarrápido
    def _nn_desde_pesos(w):
        if w.ndim == 1: w = w[None, :]
        mu = np.sum(acuerdo * w, axis=1)
        var = np.sum(w * (acuerdo - mu[:, None])**2, axis=1)
        return np.sqrt(np.maximum(mu * (1.0 - np.sqrt(var)), 0.0))

    # ---------------------------------------------------------
    # 1. NN MUESTRA
    # ---------------------------------------------------------
    w_muestra = f_t / n_total
    nn_muestra = _nn_desde_pesos(w_muestra)[0]

    # ---------------------------------------------------------
    # 2. NN POBLACIÓN (Corrección Termodinámica Exacta)
    # ---------------------------------------------------------
    counts_rounded = np.round(counts, 4)
    
    # 🚀 CORRECCIÓN 1: ORDENAR PARA AGRUPAR EN LA "FIRMA" PURA
    counts_sorted = np.sort(counts_rounded, axis=1)[:, ::-1]
    
    unique_counts, inverse_idx = np.unique(counts_sorted, axis=0, return_inverse=True)
    
    f_M = np.zeros(len(unique_counts))
    np.add.at(f_M, inverse_idx, f_t)
    f_M_mapped = f_M[inverse_idx]

    # 🚀 CORRECCIÓN 2: MASA TERMODINÁMICA COMPLETA (Jueces + Categorías)
    # A) Permutaciones de los Jueces (m! / c_i!)
    log_perm_jueces = gammaln(m_valid + 1) - np.sum(gammaln(counts + 1), axis=1)
    
    # B) Permutaciones de las Categorías (k! / f_c!)
    f_c = np.zeros((U, m + 1))
    for c in range(m + 1):
        f_c[:, c] = np.sum(counts == c, axis=1)
        
    log_asign_cat = gammaln(k_escala + 1) - np.sum(gammaln(f_c + 1), axis=1)
    
    # C) Multiplicidad Total
    log_omega = log_perm_jueces + log_asign_cat
    omega_teorico = np.exp(log_omega - np.max(log_omega))
    # ---------------------------------------------------------

    # Peso Poblacional SP
    w_pob = np.where((m_valid > 1) & (f_M_mapped > 0), f_t * (omega_teorico / f_M_mapped), 0.0)
    sum_wpob = np.sum(w_pob)
    w_pob = w_pob / sum_wpob if sum_wpob > 0 else np.ones(U) / U

    nn_poblacion = _nn_desde_pesos(w_pob)[0]

    # ---------------------------------------------------------
    # 3. IC SIMULACIÓN PONDERADA (Multinomial O(1))
    # ---------------------------------------------------------
    safe_n = max(n_total, 2)
    
    # Agrupamos los U estados en ~15 niveles únicos de acuerdo
    acuerdos_unicos, inverse_acuerdo = np.unique(np.round(acuerdo, 8), return_inverse=True)
    w_pob_grouped = np.zeros(len(acuerdos_unicos))
    np.add.at(w_pob_grouped, inverse_acuerdo, w_pob)

    # Lambda adaptada al espacio comprimido
    def _nn_desde_pesos_grouped(w_grouped):
        if w_grouped.ndim == 1: w_grouped = w_grouped[None, :]
        mu = np.sum(acuerdos_unicos * w_grouped, axis=1)
        var = np.sum(w_grouped * (acuerdos_unicos - mu[:, None])**2, axis=1)
        return np.sqrt(np.maximum(mu * (1.0 - np.sqrt(var)), 0.0))

    # Simulamos sobre el espacio enano (instantáneo)
    f_boot = np.random.multinomial(safe_n, w_pob_grouped, size=replicas)
    w_boot = f_boot / safe_n
    
    sims = _nn_desde_pesos_grouped(w_boot)

    return float(nn_muestra), float(nn_poblacion), float(np.percentile(sims, 2.5)), float(np.percentile(sims, 97.5))